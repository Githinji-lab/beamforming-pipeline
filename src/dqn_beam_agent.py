import numpy as np
import tensorflow as tf
from tensorflow import keras


class DQNBeamReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNBeamAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_tau=0.01,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=50000,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_tau = target_update_tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_net = self._build_q_network(state_dim, num_actions)
        self.target_q_net = self._build_q_network(state_dim, num_actions)
        self.target_q_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = keras.losses.Huber()
        self.replay_buffer = DQNBeamReplayBuffer(capacity=replay_capacity)

    @staticmethod
    def _build_q_network(state_dim, num_actions):
        inputs = keras.layers.Input(shape=(state_dim,))
        x = keras.layers.Dense(64, activation="relu")(inputs)
        x = keras.layers.Dense(32, activation="relu")(x)
        outputs = keras.layers.Dense(num_actions, activation="linear")(x)
        return keras.Model(inputs, outputs, name="dqn_beam_qnet")

    def act(self, state, evaluate=False):
        if (not evaluate) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(self.num_actions))
        q_vals = self.q_net(state.reshape(1, -1), training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def pretrain_imitation(self, states, target_actions, epochs=5, batch_size=32):
        y = np.zeros((len(target_actions), self.num_actions), dtype=np.float32)
        y[np.arange(len(target_actions)), target_actions] = 1.0
        self.q_net.compile(optimizer=self.optimizer, loss="mse")
        self.q_net.fit(states, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.target_q_net.set_weights(self.q_net.get_weights())

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        next_q_online = self.q_net(next_states, training=False)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

        next_q_target = self.target_q_net(next_states, training=False)
        next_q_selected = tf.gather(next_q_target, next_actions, batch_dims=1)

        targets = rewards + self.gamma * (1.0 - dones) * next_q_selected

        with tf.GradientTape() as tape:
            q_values = self.q_net(states, training=True)
            q_selected = tf.gather(q_values, actions, batch_dims=1)
            loss = self.loss_fn(targets, q_selected)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def train_on_batch(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        loss = self._train_step(states, actions, rewards, next_states, dones)
        self.soft_update_target()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return float(loss.numpy())

    def soft_update_target(self):
        q_w = self.q_net.get_weights()
        t_w = self.target_q_net.get_weights()
        new_w = [self.target_update_tau * qw + (1.0 - self.target_update_tau) * tw for qw, tw in zip(q_w, t_w)]
        self.target_q_net.set_weights(new_w)
