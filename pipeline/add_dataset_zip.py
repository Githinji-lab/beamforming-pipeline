import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, src_path)

from dataset_ingestion import ingest_dataset_zips


def _parse_zip_list(zip_values):
    parsed = []
    for item in zip_values:
        parsed.extend([part.strip() for part in item.split(",") if part.strip()])
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest dataset zip archives into data/external and register them.")
    parser.add_argument(
        "--zip",
        dest="zip_paths",
        action="append",
        required=True,
        help="Path to a .zip dataset archive. Repeat flag or pass comma-separated list.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/external",
        help="Extraction root directory.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/dataset_registry.json",
        help="Registry JSON output path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    zip_paths = _parse_zip_list(args.zip_paths)
    result = ingest_dataset_zips(
        zip_paths=zip_paths,
        output_root=args.output_root,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )

    print("=" * 72)
    print("DATASET ZIP INGESTION COMPLETE")
    print("=" * 72)
    print(f"archives_ingested={len(result['archives'])}")
    print(f"dataset_files_discovered={len(result['dataset_files'])}")
    print(f"manifest={result['manifest_path']}")

    for archive in result["archives"]:
        print(
            f"- {archive['zip_path']} -> {archive['dataset_file_count']} dataset files "
            f"({archive['extracted_file_count']} total extracted)"
        )
