#!/usr/bin/env python3
"""
Script to convert HCP mean diffusivity data to YUCCA framework format.
Converts data from extracted_mean_diffusivity_hcp_data to YUCCA task format.

Task: Task001_DMRIHCP152 (unsupervised - no labels)
Reference: https://github.com/Sllambias/yucca/blob/main/yucca/documentation/guides/task_conversion.md
"""

import json
import shutil
from pathlib import Path
import random


def create_yucca_structure(output_dir, task_name, dry_run=False):
    """
    Create the YUCCA directory structure.

    Args:
        output_dir (Path): Base output directory
        task_name (str): Task name (e.g., Task001_DMRIHCP152)
        dry_run (bool): If True, only show what would be created

    Returns:
        dict: Dictionary with paths to different directories
    """
    # Create raw_data/TaskXXX structure
    raw_data_dir = output_dir / 'raw_data'
    task_dir = raw_data_dir / task_name

    dirs = {
        'raw_data': raw_data_dir,
        'task': task_dir,
        'imagesTr': task_dir / 'imagesTr',
        'imagesTs': task_dir / 'imagesTs'
        # No labelsTr/labelsTs for unsupervised tasks
    }

    if dry_run:
        print(f"Would create YUCCA directory structure:")
        for dir_name, dir_path in dirs.items():
            print(f"  {dir_name}: {dir_path}")
        return dirs

    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")

    return dirs


def create_dataset_json(task_dir, task_name, training_cases, test_cases, dry_run=False):
    """
    Create the dataset.json file required by YUCCA.

    Args:
        task_dir (Path): Task directory path
        task_name (str): Task name
        training_cases (list): List of training case IDs
        test_cases (list): List of test case IDs
        dry_run (bool): If True, only show what would be created
    """
    dataset_info = {
        "name": task_name,
        "description": "HCP Diffusion MRI Mean Diffusivity - 152 subjects",
        "tensorImageSize": "3D",
        "reference": "Human Connectome Project",
        "licence": "HCP Open Access Data Use Terms",
        "release": "1.0",
        "modality": {
            "0": "MD"  # Mean Diffusivity
        },
        "labels": {},  # No labels for unsupervised task
        "numTraining": len(training_cases),
        "numTest": len(test_cases),
        "training": [
            {
                "image": f"./imagesTr/{case_id}_000.nii.gz"
            }
            for case_id in training_cases
        ],
        "test": [
            f"./imagesTs/{case_id}_000.nii.gz"
            for case_id in test_cases
        ]
    }

    dataset_json_path = task_dir / 'dataset.json'

    if dry_run:
        print(f"\nWould create dataset.json at: {dataset_json_path}")
        print(f"  Training cases: {len(training_cases)}")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Sample training entry: {dataset_info['training'][0] if training_cases else 'None'}")
        print(f"  Sample test entry: {dataset_info['test'][0] if test_cases else 'None'}")
        return dataset_json_path

    # Write dataset.json
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nCreated dataset.json: {dataset_json_path}")
    print(f"  Training cases: {len(training_cases)}")
    print(f"  Test cases: {len(test_cases)}")

    return dataset_json_path


def copy_images(patient_dirs, yucca_dirs, training_cases, test_cases, dry_run=False):
    """
    Copy mean diffusivity images to YUCCA format directories.

    Args:
        patient_dirs (dict): Dictionary mapping patient_id to source directory
        yucca_dirs (dict): Dictionary with YUCCA directory paths
        training_cases (list): List of training case IDs
        test_cases (list): List of test case IDs
        dry_run (bool): If True, only show what would be copied
    """
    copied_count = 0
    failed_count = 0

    # Process training cases
    print(f"\nProcessing training cases...")
    for case_id in training_cases:
        if case_id not in patient_dirs:
            print(f"  Warning: Patient {case_id} not found in source data")
            failed_count += 1
            continue

        source_file = patient_dirs[case_id] / 'mean_diffusivity.nii.gz'
        target_file = yucca_dirs['imagesTr'] / f"{case_id}_000.nii.gz"

        if not source_file.exists():
            print(f"  Error: Source file not found: {source_file}")
            failed_count += 1
            continue

        if dry_run:
            print(f"  Would copy: {source_file.name} -> {target_file.name}")
        else:
            try:
                shutil.copy2(source_file, target_file)
                print(f"  Copied: {case_id} -> training set")
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {case_id}: {e}")
                failed_count += 1

    # Process test cases
    print(f"\nProcessing test cases...")
    for case_id in test_cases:
        if case_id not in patient_dirs:
            print(f"  Warning: Patient {case_id} not found in source data")
            failed_count += 1
            continue

        source_file = patient_dirs[case_id] / 'mean_diffusivity.nii.gz'
        target_file = yucca_dirs['imagesTs'] / f"{case_id}_000.nii.gz"

        if not source_file.exists():
            print(f"  Error: Source file not found: {source_file}")
            failed_count += 1
            continue

        if dry_run:
            print(f"  Would copy: {source_file.name} -> {target_file.name}")
        else:
            try:
                shutil.copy2(source_file, target_file)
                print(f"  Copied: {case_id} -> test set")
                copied_count += 1
            except Exception as e:
                print(f"  Error copying {case_id}: {e}")
                failed_count += 1

    return copied_count, failed_count


def convert_to_yucca(data_dir="extracted_mean_diffusivity_hcp_data",
                     output_dir="yucca_tasks",
                     task_name="Task001_DMRIHCP152",
                     test_set_amount=0,
                     random_seed=42,
                     dry_run=False):
    """
    Convert HCP mean diffusivity data to YUCCA framework format.

    Args:
        data_dir (str): Directory containing patient subdirectories with mean_diffusivity.nii.gz
        output_dir (str): Output directory for YUCCA tasks
        task_name (str): Name of the YUCCA task
        test_set_amount (int): Number of cases to use for test set
        random_seed (int): Random seed for reproducible train/test split
        dry_run (bool): If True, only show what would be done

    Returns:
        bool: True if successful, False if failed
    """
    # Get the script directory and construct paths
    script_dir = Path(__file__).parent
    data_path = script_dir / data_dir
    output_path = script_dir / output_dir

    if not data_path.exists():
        print(f"Error: Data directory '{data_path}' does not exist!")
        return False

    # Find all patient directories with mean_diffusivity.nii.gz
    patient_dirs = {}
    for patient_dir in data_path.iterdir():
        if patient_dir.is_dir():
            md_file = patient_dir / 'mean_diffusivity.nii.gz'
            if md_file.exists():
                patient_dirs[patient_dir.name] = patient_dir

    if not patient_dirs:
        print(f"No patient directories with mean_diffusivity.nii.gz found in '{data_path}'")
        return False

    total_patients = len(patient_dirs)
    patient_ids = sorted(patient_dirs.keys())

    print(f"Found {total_patients} patients with mean diffusivity data")

    # Validate test set amount
    if test_set_amount < 0:
        print(f"Error: test_set_amount must be non-negative, got {test_set_amount}")
        return False

    if test_set_amount > total_patients:
        print(f"Error: test_set_amount ({test_set_amount}) cannot exceed total patients ({total_patients})")
        return False

    # Split into train/test sets
    random.seed(random_seed)
    shuffled_ids = patient_ids.copy()
    random.shuffle(shuffled_ids)

    test_cases = shuffled_ids[:test_set_amount]
    training_cases = shuffled_ids[test_set_amount:]

    print(f"\nDataset split (random_seed={random_seed}):")
    print(f"  Training cases: {len(training_cases)}")
    print(f"  Test cases: {len(test_cases)}")

    if dry_run:
        print(f"\nDry run mode - showing what would be created:")
        print("=" * 70)
    else:
        print(f"\nConverting to YUCCA format...")
        print("=" * 70)

    # Create YUCCA directory structure
    yucca_dirs = create_yucca_structure(output_path, task_name, dry_run)

    # Create dataset.json
    dataset_json_path = create_dataset_json(
        yucca_dirs['task'], task_name, training_cases, test_cases, dry_run
    )

    # Copy images
    copied_count, failed_count = copy_images(
        patient_dirs, yucca_dirs, training_cases, test_cases, dry_run
    )

    print("\n" + "=" * 70)
    print(f"Conversion completed:")
    if not dry_run:
        print(f"  - Successfully copied: {copied_count} files")
        if failed_count > 0:
            print(f"  - Failed to copy: {failed_count} files")
        print(f"  - Task directory: {yucca_dirs['task']}")
        print(f"  - Dataset JSON: {dataset_json_path}")
    else:
        print(f"  - Would process: {len(training_cases) + len(test_cases)} files")
        print(f"  - Would create task: {task_name}")

    return failed_count == 0


def main():
    """Main function to handle command line arguments and run the conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HCP mean diffusivity data to YUCCA framework format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    parser.add_argument(
        "--data-dir",
        default="extracted_mean_diffusivity_hcp_data",
        help="Directory containing patient subdirectories with mean_diffusivity.nii.gz (default: extracted_mean_diffusivity_hcp_data)"
    )
    parser.add_argument(
        "--output-dir",
        default="yucca_tasks",
        help="Output directory for YUCCA tasks (default: yucca_tasks)"
    )
    parser.add_argument(
        "--task-name",
        default="Task001_DMRIHCP152",
        help="Name of the YUCCA task (default: Task001_DMRIHCP152)"
    )
    parser.add_argument(
        "--test-set-amount",
        type=int,
        default=0,
        help="Number of cases to use for test set (default: 0 - all for training)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split (default: 42)"
    )

    args = parser.parse_args()

    # Run the conversion
    success = convert_to_yucca(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task_name=args.task_name,
        test_set_amount=args.test_set_amount,
        random_seed=args.random_seed,
        dry_run=args.dry_run
    )

    if not success:
        print("Conversion failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
