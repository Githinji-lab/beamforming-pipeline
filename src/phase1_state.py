import numpy as np


class Phase1StateAugmenter:
    def __init__(self, enabled=False, num_clusters=0):
        self.enabled = bool(enabled)
        self.num_clusters = int(max(0, num_clusters))
        self.centroids = None

    def fit(self, H_samples):
        if not self.enabled or self.num_clusters <= 1:
            self.centroids = None
            return self

        features = np.array([self._cluster_feature(h) for h in H_samples], dtype=np.float32)
        self.centroids = self._kmeans(features, self.num_clusters)
        return self

    def transform(self, base_state, H, snr, prev_H=None):
        base_state = np.array(base_state, dtype=np.float32).reshape(-1)
        if not self.enabled:
            return base_state

        engineered = self._engineered_features(H=H, snr=snr, prev_H=prev_H)
        cluster_features = self._cluster_one_hot(H)

        return np.concatenate([base_state, engineered, cluster_features], axis=0).astype(np.float32)

    def _cluster_feature(self, H):
        mags = np.abs(H).reshape(-1)
        return np.array([
            float(np.mean(mags)),
            float(np.std(mags)),
            float(np.percentile(mags, 75)),
            float(np.percentile(mags, 90)),
        ], dtype=np.float32)

    def _cluster_one_hot(self, H):
        if self.centroids is None:
            return np.zeros(0, dtype=np.float32)

        feature = self._cluster_feature(H)
        distances = np.linalg.norm(self.centroids - feature.reshape(1, -1), axis=1)
        idx = int(np.argmin(distances))
        one_hot = np.zeros(self.centroids.shape[0], dtype=np.float32)
        one_hot[idx] = 1.0
        return one_hot

    def _engineered_features(self, H, snr, prev_H=None):
        mags = np.abs(H)
        phases = np.angle(H)

        hh = H @ H.conj().T
        try:
            cond = float(np.linalg.cond(hh + 1e-8 * np.eye(hh.shape[0])))
        except Exception:
            cond = 1.0

        mobility = 0.0
        if prev_H is not None:
            denom = np.linalg.norm(prev_H) + 1e-9
            mobility = float(np.linalg.norm(H - prev_H) / denom)

        feats = np.array(
            [
                float(np.mean(mags)),
                float(np.std(mags)),
                float(np.max(mags)),
                float(np.var(phases)),
                float(np.log1p(cond)),
                float(snr),
                mobility,
            ],
            dtype=np.float32,
        )
        return feats

    @staticmethod
    def _kmeans(x, k, iterations=30, seed=42):
        rng = np.random.RandomState(seed)
        if x.shape[0] <= k:
            return x[:k].copy()

        indices = rng.choice(x.shape[0], size=k, replace=False)
        centroids = x[indices].copy()

        for _ in range(iterations):
            distances = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = x[mask].mean(axis=0)
            if np.allclose(new_centroids, centroids, atol=1e-5):
                break
            centroids = new_centroids

        return centroids
