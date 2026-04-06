import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile


SUPPORTED_EXTENSIONS = {".mat", ".csv", ".pkl", ".npy", ".npz"}


def _sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(zip_path, destination_dir, overwrite=False):
    extracted_files = []
    with ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue

            normalized = os.path.normpath(member.filename)
            if normalized.startswith("..") or os.path.isabs(normalized):
                raise ValueError(f"Unsafe path in zip '{zip_path}': {member.filename}")

            target_path = os.path.join(destination_dir, normalized)
            target_parent = os.path.dirname(target_path)
            os.makedirs(target_parent, exist_ok=True)

            if os.path.exists(target_path) and not overwrite:
                extracted_files.append(target_path)
                continue

            with zf.open(member, "r") as source, open(target_path, "wb") as target:
                target.write(source.read())

            extracted_files.append(target_path)
    return extracted_files


def ingest_dataset_zips(zip_paths, output_root="data/external", manifest_path="data/dataset_registry.json", overwrite=False):
    if not zip_paths:
        return {"archives": [], "dataset_files": [], "manifest_path": manifest_path}

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    archives_summary = []
    discovered_dataset_files = []

    for zip_item in zip_paths:
        zip_path = Path(zip_item).expanduser().resolve()
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip archive not found: {zip_item}")
        if zip_path.suffix.lower() != ".zip":
            raise ValueError(f"Expected .zip archive, got: {zip_item}")

        archive_stem = zip_path.stem
        archive_out_dir = output_root_path / archive_stem
        archive_out_dir.mkdir(parents=True, exist_ok=True)

        extracted = _safe_extract(str(zip_path), str(archive_out_dir), overwrite=overwrite)
        dataset_files = [
            f for f in extracted if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        discovered_dataset_files.extend(dataset_files)

        archives_summary.append(
            {
                "zip_path": str(zip_path),
                "zip_sha256": _sha256_file(str(zip_path)),
                "extracted_dir": str(archive_out_dir.resolve()),
                "extracted_file_count": len(extracted),
                "dataset_file_count": len(dataset_files),
                "dataset_files": sorted(str(Path(f).resolve()) for f in dataset_files),
            }
        )

    registry = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "output_root": str(output_root_path.resolve()),
        "archives": archives_summary,
        "dataset_files": sorted(str(Path(f).resolve()) for f in discovered_dataset_files),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }

    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump(registry, f, indent=2)

    registry["manifest_path"] = str(manifest.resolve())
    return registry
