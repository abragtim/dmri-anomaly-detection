#!/usr/bin/env python3
"""
Script to create an oval lesion in a DMRI image for patient 2218031

This script adds an artificial oval lesion around specified coordinates
by setting pixel values to an anomalous value.
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def create_oval_mask(shape, center, semi_axes):
    """
    Create a 3D oval (ellipsoid) mask.

    Args:
        shape (tuple): Shape of the 3D volume (z, y, x)
        center (tuple): Center coordinates (z, y, x)
        semi_axes (tuple): Semi-axes lengths (a, b, c) for the ellipsoid

    Returns:
        np.ndarray: Boolean mask of the ellipsoid
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Ellipsoid equation: (x-cx)²/a² + (y-cy)²/b² + (z-cz)²/c² <= 1
    cx, cy, cz = center
    a, b, c = semi_axes

    ellipsoid = ((x - cz)**2 / c**2 +
                 (y - cy)**2 / b**2 +
                 (z - cx)**2 / a**2) <= 1

    return ellipsoid


def add_oval_lesion(image_path, output_path, center, semi_axes, lesion_value=None):
    """
    Add an oval lesion to a NIfTI image.

    Args:
        image_path (str): Path to input NIfTI file
        output_path (str): Path to save modified NIfTI file
        center (tuple): Center coordinates (z, y, x)
        semi_axes (tuple): Semi-axes lengths (a, b, c)
        lesion_value (float): Value to set for lesion pixels. If None, uses 3x the mean intensity

    Returns:
        dict: Statistics about the lesion
    """
    # Load the NIfTI image
    nii_img = nib.load(image_path)
    data = nii_img.get_fdata()

    print(f"Image shape: {data.shape}")
    print(f"Original data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"Original data mean: {data.mean():.4f}")

    # Create a copy to modify
    modified_data = data.copy()

    # Create oval mask
    mask = create_oval_mask(data.shape, center, semi_axes)

    # Calculate lesion value if not provided
    if lesion_value is None:
        # Set lesion to 3 times the mean intensity (clearly anomalous)
        lesion_value = 3 * data.mean()

    # Apply the lesion
    original_values = modified_data[mask]
    modified_data[mask] = lesion_value

    # Create new NIfTI image with the same header and affine
    modified_nii = nib.Nifti1Image(modified_data, nii_img.affine, nii_img.header)

    # Save the modified image
    nib.save(modified_nii, output_path)

    # Calculate statistics
    num_lesion_pixels = np.sum(mask)
    lesion_volume_voxels = num_lesion_pixels

    stats = {
        'center': center,
        'semi_axes': semi_axes,
        'lesion_value': lesion_value,
        'num_lesion_pixels': num_lesion_pixels,
        'lesion_volume_voxels': lesion_volume_voxels,
        'original_mean_in_lesion': original_values.mean(),
        'original_std_in_lesion': original_values.std(),
        'image_shape': data.shape
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Add oval lesion to DMRI image')
    parser.add_argument('--patient_id', default='2218031',
                       help='Patient ID (default: 2218031)')
    parser.add_argument('--center', nargs=3, type=int, default=[50, 195, 328],
                       help='Center coordinates (z y x) (default: 50 195 328)')
    parser.add_argument('--semi_axes', nargs=3, type=int, default=[8, 12, 10],
                       help='Semi-axes lengths (a b c) (default: 8 12 10)')
    parser.add_argument('--lesion_value', type=float, default=None,
                       help='Lesion intensity value (default: 3x mean intensity)')
    parser.add_argument('--input_dir', default='yucca_tasks/raw_data/Task001_DMRIHCP152/imagesTs',
                       help='Input directory containing test images')
    parser.add_argument('--output_dir', default='lesion_modified_images',
                       help='Output directory for modified images')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find the patient image
    patient_file = f"{args.patient_id}_000.nii.gz"
    input_path = input_dir / patient_file
    output_path = output_dir / f"{args.patient_id}_lesion.nii.gz"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Processing patient {args.patient_id}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Lesion center (z,y,x): {args.center}")
    print(f"Semi-axes (a,b,c): {args.semi_axes}")

    # Add the lesion
    try:
        stats = add_oval_lesion(
            str(input_path),
            str(output_path),
            tuple(args.center),
            tuple(args.semi_axes),
            args.lesion_value
        )

        print("\n=== Lesion Creation Results ===")
        print(f"✓ Successfully created lesion for patient {args.patient_id}")
        print(f"  Center coordinates: {stats['center']}")
        print(f"  Semi-axes: {stats['semi_axes']}")
        print(f"  Lesion value: {stats['lesion_value']:.4f}")
        print(f"  Number of lesion pixels: {stats['num_lesion_pixels']}")
        print(f"  Original mean in lesion area: {stats['original_mean_in_lesion']:.4f}")
        print(f"  Original std in lesion area: {stats['original_std_in_lesion']:.4f}")
        print(f"  Image shape: {stats['image_shape']}")
        print(f"  Modified image saved to: {output_path}")

        # Save statistics to text file
        stats_file = output_dir / f"{args.patient_id}_lesion_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("Oval Lesion Creation Statistics\n")
            f.write("=" * 35 + "\n")
            f.write(f"Patient ID: {args.patient_id}\n")
            f.write(f"Center coordinates (z,y,x): {stats['center']}\n")
            f.write(f"Semi-axes (a,b,c): {stats['semi_axes']}\n")
            f.write(f"Lesion value: {stats['lesion_value']:.4f}\n")
            f.write(f"Number of lesion pixels: {stats['num_lesion_pixels']}\n")
            f.write(f"Lesion volume (voxels): {stats['lesion_volume_voxels']}\n")
            f.write(f"Original mean in lesion area: {stats['original_mean_in_lesion']:.4f}\n")
            f.write(f"Original std in lesion area: {stats['original_std_in_lesion']:.4f}\n")
            f.write(f"Image shape: {stats['image_shape']}\n")
            f.write(f"Input file: {input_path}\n")
            f.write(f"Output file: {output_path}\n")

        print(f"  Statistics saved to: {stats_file}")

    except Exception as e:
        print(f"Error creating lesion: {e}")
        return


if __name__ == "__main__":
    main()
