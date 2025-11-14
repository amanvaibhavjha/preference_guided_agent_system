#!/usr/bin/env python3
"""
Download CORAL datasets from HuggingFace.

Usage:
    python scripts/download_coral.py
"""

import os
import json
from pathlib import Path
from datasets import load_dataset


def download_coral_datasets(output_dir="data/coral"):
    """
    Download CORAL datasets from HuggingFace.

    Args:
        output_dir: Directory to save downloaded datasets
    """
    print("=" * 60)
    print("CORAL Dataset Downloader")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Datasets to download
    datasets_to_download = ["pearl", "inspired", "redial"]

    for dataset_name in datasets_to_download:
        print(f"\n📥 Downloading {dataset_name}...")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset("kookeej/CORAL", dataset_name)

            # Save each split
            for split in dataset.keys():
                split_output_dir = output_path / dataset_name
                split_output_dir.mkdir(parents=True, exist_ok=True)

                output_file = split_output_dir / f"{split}.jsonl"

                print(f"   💾 Saving {split} split to {output_file}")

                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in dataset[split]:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')

                print(f"   ✓ Saved {len(dataset[split])} samples")

            print(f"✓ {dataset_name} downloaded successfully")

        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {str(e)}")
            continue

    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print(f"📁 Data saved to: {output_path.absolute()}")
    print("=" * 60)

    # Print summary
    print("\n📊 Dataset Summary:")
    for dataset_name in datasets_to_download:
        dataset_dir = output_path / dataset_name
        if dataset_dir.exists():
            files = list(dataset_dir.glob("*.jsonl"))
            print(f"\n{dataset_name}:")
            for file in files:
                line_count = sum(1 for _ in open(file, 'r', encoding='utf-8'))
                print(f"  - {file.name}: {line_count} samples")


def inspect_sample(output_dir="data/coral"):
    """
    Inspect a sample from each dataset to understand the structure.

    Args:
        output_dir: Directory where datasets are saved
    """
    print("\n" + "=" * 60)
    print("🔍 Dataset Structure Inspection")
    print("=" * 60)

    datasets_to_inspect = ["pearl", "inspired", "redial"]

    for dataset_name in datasets_to_inspect:
        train_file = Path(output_dir) / dataset_name / "train.jsonl"

        if not train_file.exists():
            print(f"\n⚠️  {dataset_name} not found")
            continue

        print(f"\n📋 {dataset_name.upper()} Sample:")
        print("-" * 60)

        with open(train_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)

            # Pretty print with indentation
            print(json.dumps(sample, indent=2, ensure_ascii=False))

        print("-" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CORAL datasets")
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/coral',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect dataset structure after download'
    )

    args = parser.parse_args()

    # Download datasets
    download_coral_datasets(args.output_dir)

    # Optionally inspect
    if args.inspect:
        inspect_sample(args.output_dir)

    print("\n🎉 All done! Next steps:")
    print("1. Inspect the data: jupyter notebook notebooks/data_exploration.ipynb")
    print("2. Preprocess data: python scripts/preprocess_data.py")
    print("3. Start training: python scripts/train_preference_model.py")
