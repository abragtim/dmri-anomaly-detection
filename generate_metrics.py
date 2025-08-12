#!/usr/bin/env python3
"""
Script to generate DTI metrics (mean diffusivity) for each patient in extracted_hcp_data.
Uses DIPY to process diffusion data and compute mean diffusivity maps.
"""

import os
import sys
from pathlib import Path
import traceback

import nibabel as nib
from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io import read_bvals_bvecs
from dipy.reconst.dti import TensorModel
import numpy as np


def process_patient(patient_dir, output_dir=None, dry_run=False):
    """
    Process a single patient directory to generate mean diffusivity.

    Args:
        patient_dir (Path): Path to patient directory containing diffusion data
        output_dir (Path): Optional output directory (defaults to patient_dir)
        dry_run (bool): If True, only validate files without processing

    Returns:
        bool: True if successful, False if failed
    """
    patient_id = patient_dir.name

    # Define required files
    required_files = {
        'data': patient_dir / 'data.nii.gz',
        'bvals': patient_dir / 'bvals',
        'bvecs': patient_dir / 'bvecs',
        'mask': patient_dir / 'nodif_brain_mask.nii.gz'
    }

    # Check if all required files exist
    missing_files = []
    for file_type, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append(f"{file_type}: {file_path.name}")

    if missing_files:
        print(f"  Error: Missing files: {', '.join(missing_files)}")
        return False

    # Set output directory
    if output_dir is None:
        output_dir = patient_dir
    else:
        output_dir = output_dir / patient_id
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'mean_diffusivity.nii.gz'

    if dry_run:
        print(f"  Would process: {patient_id}")
        print(f"    Input files: {len(required_files)} files found")
        print(f"    Output: {output_file}")
        return True

    try:
        print(f"  Loading diffusion data...")

        # Load diffusion data
        img = nib.load(str(required_files['data']))
        data = img.get_fdata()
        affine = img.affine

        print(f"    Data shape: {data.shape}")

        # Load bvals and bvecs
        bvals, bvecs = read_bvals_bvecs(str(required_files['bvals']), str(required_files['bvecs']))
        gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)

        print(f"    Gradient table: {len(bvals)} directions")

        # Load brain mask
        mask = nib.load(str(required_files['mask'])).get_fdata().astype(bool)

        print(f"    Brain mask: {np.sum(mask)} voxels")

        # Fit DTI model
        print(f"  Fitting DTI model...")
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask=mask)

        # Calculate mean diffusivity
        print(f"  Computing mean diffusivity...")
        mean_diffusivity = tenfit.md

        # Save result
        print(f"  Saving: {output_file.name}")
        nib.save(nib.Nifti1Image(mean_diffusivity, affine), str(output_file))

        print(f"  ✓ Successfully processed {patient_id}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {patient_id}: {str(e)}")
        if isinstance(e, (MemoryError, OSError)):
            print(f"    This might be due to insufficient memory or disk space")
        else:
            print(f"    Full error: {traceback.format_exc()}")
        return False


def generate_metrics(data_dir="extracted_hcp_data", output_dir=None, dry_run=False, max_patients=None):
    """
    Generate DTI metrics for all patients in the data directory.

    Args:
        data_dir (str): Directory containing patient subdirectories
        output_dir (str): Optional output directory (defaults to data_dir)
        dry_run (bool): If True, only validate files without processing
        max_patients (int): Optional limit on number of patients to process

    Returns:
        bool: True if all patients processed successfully
    """
    # Get the script directory and construct paths
    script_dir = Path(__file__).parent
    data_path = script_dir / data_dir

    if not data_path.exists():
        print(f"Error: Directory '{data_path}' does not exist!")
        return False

    if not data_path.is_dir():
        print(f"Error: '{data_path}' is not a directory!")
        return False

    # Set output directory
    output_path = None
    if output_dir:
        output_path = script_dir / output_dir
        if not dry_run:
            output_path.mkdir(exist_ok=True)

    # Find all patient directories
    patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    patient_dirs.sort()  # Sort for consistent processing order

    if not patient_dirs:
        print(f"No patient directories found in '{data_path}'")
        return False

    # Limit number of patients if specified
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
        print(f"Processing first {len(patient_dirs)} patients (limited by max_patients={max_patients})")

    print(f"Found {len(patient_dirs)} patient directories to process")

    if dry_run:
        print("\nDry run mode - validating files and showing what would be processed:")
        print("=" * 70)
    else:
        print(f"\nProcessing patients to generate mean diffusivity maps...")
        if output_path:
            print(f"Output directory: {output_path}")
        else:
            print(f"Output: in-place (same as input directories)")
        print("=" * 70)

    successful_count = 0
    failed_count = 0

    for i, patient_dir in enumerate(patient_dirs, 1):
        print(f"\n[{i}/{len(patient_dirs)}] Processing patient: {patient_dir.name}")

        success = process_patient(patient_dir, output_path, dry_run)

        if success:
            successful_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 70)
    print(f"Processing completed:")
    print(f"  - Successfully processed: {successful_count} patients")
    if failed_count > 0:
        print(f"  - Failed to process: {failed_count} patients")

    if not dry_run and successful_count > 0:
        if output_path:
            print(f"\nGenerated mean diffusivity maps saved to: {output_path}")
        else:
            print(f"\nGenerated mean diffusivity maps saved in patient directories")

    return failed_count == 0


def main():
    """Main function to handle command line arguments and run the processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DTI metrics (mean diffusivity) for HCP diffusion data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and show what would be processed without actually processing"
    )
    parser.add_argument(
        "--data-dir",
        default="extracted_hcp_data",
        help="Directory containing patient subdirectories (default: extracted_hcp_data)"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save processed files to (default: in-place in patient directories)"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Maximum number of patients to process (useful for testing)"
    )

    args = parser.parse_args()

    print("DTI Metrics Generation Script")
    print("=" * 70)
    print("Generates: mean_diffusivity.nii.gz for each patient")

    if args.dry_run:
        print("Running in DRY RUN mode - no processing will be performed")
    print("=" * 70)

    # Check if required packages are available
    try:
        import nibabel as nib
        from dipy.core.gradients import gradient_table_from_bvals_bvecs
        from dipy.io import read_bvals_bvecs
        from dipy.reconst.dti import TensorModel
        print("✓ All required packages (nibabel, dipy) are available")
    except ImportError as e:
        print(f"✗ Missing required packages: {e}")
        print("Please install with: pip install nibabel dipy numpy")
        sys.exit(1)

    # Process patients
    success = generate_metrics(
        args.data_dir,
        args.output_dir,
        args.dry_run,
        args.max_patients
    )

    if not success:
        print("Processing failed!")
        sys.exit(1)

    if not args.dry_run:
        print("\nProcessing completed successfully!")
    else:
        print("\nDry run completed - ready for actual processing!")


if __name__ == "__main__":
    main()
