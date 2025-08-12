#!/usr/bin/env python3
"""
Script to extract HCP diffusion data from zip files and keep only essential files.
Extracts: bvals, bvecs, data.nii.gz, grad_dev.nii.gz, nodif_brain_mask.nii.gz for each patient.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path


def extract_hcp_data(data_dir="hcp_data", output_dir="extracted_data", dry_run=False):
    """
    Extract HCP data from zip files and keep only essential diffusion files.

    Args:
        data_dir (str): Directory containing zip files
        output_dir (str): Directory to extract files to
        dry_run (bool): If True, only show what would be extracted without actually extracting
    """
    # Get the script directory and construct paths
    script_dir = Path(__file__).parent
    data_path = script_dir / data_dir
    output_path = script_dir / output_dir

    if not data_path.exists():
        print(f"Error: Directory '{data_path}' does not exist!")
        return False

    # Find all zip files
    zip_files = list(data_path.glob("*.zip"))
    if not zip_files:
        print(f"No zip files found in '{data_path}'")
        return False

    print(f"Found {len(zip_files)} zip files to process")

    # Files we want to keep
    target_files = {'bvals', 'bvecs', 'data.nii.gz', 'grad_dev.nii.gz', 'nodif_brain_mask.nii.gz'}

    if dry_run:
        print("\nDry run mode - showing what would be extracted:")
        print("=" * 60)
    else:
        # Create output directory
        output_path.mkdir(exist_ok=True)
        print(f"\nExtracting to: {output_path}")
        print("=" * 60)

    processed_count = 0
    failed_count = 0

    for zip_file in sorted(zip_files):
        patient_id = zip_file.stem.split('_')[0]  # Extract patient ID from filename
        print(f"\nProcessing: {zip_file.name} (Patient: {patient_id})")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Find the diffusion directory in the zip
                diffusion_files = [f for f in zf.namelist() if 'T1w/Diffusion/' in f and not f.endswith('/')]

                if not diffusion_files:
                    print(f"  Warning: No diffusion files found in {zip_file.name}")
                    continue

                # Filter for target files
                files_to_extract = []
                for file_path in diffusion_files:
                    filename = Path(file_path).name
                    if filename in target_files:
                        files_to_extract.append(file_path)

                if len(files_to_extract) != len(target_files):
                    found_files = {Path(f).name for f in files_to_extract}
                    missing_files = target_files - found_files
                    print(f"  Warning: Missing files: {missing_files}")

                if not files_to_extract:
                    print(f"  Error: No target files found in {zip_file.name}")
                    failed_count += 1
                    continue

                if dry_run:
                    print(f"  Would extract {len(files_to_extract)} files:")
                    for file_path in files_to_extract:
                        filename = Path(file_path).name
                        print(f"    - {filename}")
                else:
                    # Create patient directory
                    patient_dir = output_path / patient_id
                    patient_dir.mkdir(exist_ok=True)

                    # Extract target files
                    extracted_files = []
                    for file_path in files_to_extract:
                        filename = Path(file_path).name

                        # Extract file to temporary location
                        with zf.open(file_path) as source:
                            target_file = patient_dir / filename
                            with open(target_file, 'wb') as target:
                                shutil.copyfileobj(source, target)

                        extracted_files.append(filename)

                    print(f"  Extracted {len(extracted_files)} files: {', '.join(extracted_files)}")

                processed_count += 1

        except Exception as e:
            print(f"  Error processing {zip_file.name}: {e}")
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"Processing completed:")
    print(f"  - Successfully processed: {processed_count} files")
    if failed_count > 0:
        print(f"  - Failed to process: {failed_count} files")

    if not dry_run and processed_count > 0:
        print(f"\nExtracted data saved to: {output_path}")
        print(f"Each patient directory contains: {', '.join(target_files)}")

    return failed_count == 0


def cleanup_zip_files(data_dir="hcp_data", dry_run=False):
    """
    Remove zip files after successful extraction.

    Args:
        data_dir (str): Directory containing zip files
        dry_run (bool): If True, only show what would be deleted
    """
    script_dir = Path(__file__).parent
    data_path = script_dir / data_dir

    zip_files = list(data_path.glob("*.zip"))
    if not zip_files:
        print("No zip files found to clean up.")
        return True

    print(f"\nFound {len(zip_files)} zip files to remove:")
    for zip_file in zip_files:
        print(f"  - {zip_file.name}")

    if dry_run:
        print("\nDry run mode - no files would be deleted.")
        return True

    response = input(f"\nAre you sure you want to delete these {len(zip_files)} zip files? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cleanup cancelled.")
        return False

    removed_count = 0
    for zip_file in zip_files:
        try:
            zip_file.unlink()
            print(f"Removed: {zip_file.name}")
            removed_count += 1
        except OSError as e:
            print(f"Failed to remove {zip_file.name}: {e}")

    print(f"\nRemoved {removed_count} zip files.")
    return True


def main():
    """Main function to handle command line arguments and run the extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract essential diffusion files from HCP zip archives"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without actually extracting files"
    )
    parser.add_argument(
        "--data-dir",
        default="hcp_data",
        help="Directory containing zip files (default: hcp_data)"
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_data",
        help="Directory to extract files to (default: extracted_data)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove zip files after successful extraction"
    )

    args = parser.parse_args()

    print("HCP Data Extraction Script")
    print("=" * 60)
    print(f"Target files: bvals, bvecs, data.nii.gz, grad_dev.nii.gz, nodif_brain_mask.nii.gz")

    if args.dry_run:
        print("Running in DRY RUN mode - no files will be extracted")
    print("=" * 60)

    # Extract data
    success = extract_hcp_data(args.data_dir, args.output_dir, args.dry_run)

    if not success:
        print("Extraction failed!")
        sys.exit(1)

    # Cleanup zip files if requested
    if args.cleanup and not args.dry_run:
        cleanup_success = cleanup_zip_files(args.data_dir, args.dry_run)
        if not cleanup_success:
            print("Cleanup failed!")
            sys.exit(1)

    print("\nExtraction completed successfully!")


if __name__ == "__main__":
    main()
