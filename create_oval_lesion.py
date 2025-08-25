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


def create_square_mask(shape, center, size, offset_z=0):
    """
    Create a 3D square (cubic) mask positioned relative to a center point.

    Args:
        shape (tuple): Shape of the 3D volume (z, y, x)
        center (tuple): Reference center coordinates (z, y, x)
        size (tuple): Size of the square mask (depth, height, width)
        offset_z (int): Offset in z-direction from center (negative = above, positive = below)

    Returns:
        np.ndarray: Boolean mask of the square region
    """
    cx, cy, cz = center
    depth, height, width = size

    # Calculate square bounds with offset
    z_center = cx + offset_z
    z_min = max(0, z_center - depth // 2)
    z_max = min(shape[0], z_center + depth // 2)
    y_min = max(0, cy - height // 2)
    y_max = min(shape[1], cy + height // 2)
    x_min = max(0, cz - width // 2)
    x_max = min(shape[2], cz + width // 2)

    # Create the mask
    mask = np.zeros(shape, dtype=bool)
    mask[z_min:z_max, y_min:y_max, x_min:x_max] = True

    return mask


def add_oval_lesion(image_path, output_path, center, semi_axes, lesion_value=None,
                   add_hiding_square=False, square_size=None, square_offset=-15):
    """
    Add an oval lesion to a NIfTI image with optional hiding square mask.

    Args:
        image_path (str): Path to input NIfTI file
        output_path (str): Path to save modified NIfTI file
        center (tuple): Center coordinates (z, y, x)
        semi_axes (tuple): Semi-axes lengths (a, b, c)
        lesion_value (float): Value to set for lesion pixels. If None, uses 3x the mean intensity
        add_hiding_square (bool): Whether to add a square mask with 0 values above the oval
        square_size (tuple): Size of hiding square (depth, height, width). If None, auto-calculated
        square_offset (int): Z-offset for square position relative to oval center (negative = above)

    Returns:
        dict: Statistics about the lesion and hiding square
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
    oval_mask = create_oval_mask(data.shape, center, semi_axes)

    # Calculate lesion value if not provided
    if lesion_value is None:
        # Set lesion to 3 times the mean intensity (clearly anomalous)
        lesion_value = 3 * data.mean()

    # Apply the oval lesion
    original_oval_values = modified_data[oval_mask]
    modified_data[oval_mask] = lesion_value

    # Initialize square mask variables
    square_mask = None
    original_square_values = None

    # Add hiding square if requested
    if add_hiding_square:
        # Auto-calculate square size if not provided (make it larger than the oval)
        if square_size is None:
            # Make square 1.5x larger than oval in each dimension
            square_size = (
                int(semi_axes[0] * 3),  # depth
                int(semi_axes[1] * 3),  # height
                int(semi_axes[2] * 3)   # width
            )

        print(f"Adding hiding square with size: {square_size}")
        print(f"Square offset from oval center: {square_offset}")

        # Create square mask
        square_mask = create_square_mask(data.shape, center, square_size, square_offset)

        # Store original values in square area
        original_square_values = modified_data[square_mask]

        # Set square area to 0 (hiding the area)
        modified_data[square_mask] = 0.0

    # Create new NIfTI image with the same header and affine
    modified_nii = nib.Nifti1Image(modified_data, nii_img.affine, nii_img.header)

    # Save the modified image
    nib.save(modified_nii, output_path)

    # Calculate statistics
    num_oval_pixels = np.sum(oval_mask)
    num_square_pixels = np.sum(square_mask) if square_mask is not None else 0

    stats = {
        'center': center,
        'semi_axes': semi_axes,
        'lesion_value': lesion_value,
        'num_oval_pixels': num_oval_pixels,
        'oval_volume_voxels': num_oval_pixels,
        'original_mean_in_oval': original_oval_values.mean(),
        'original_std_in_oval': original_oval_values.std(),
        'image_shape': data.shape,
        'hiding_square_added': add_hiding_square,
        'square_size': square_size if add_hiding_square else None,
        'square_offset': square_offset if add_hiding_square else None,
        'num_square_pixels': num_square_pixels,
        'original_mean_in_square': original_square_values.mean() if original_square_values is not None else None,
        'original_std_in_square': original_square_values.std() if original_square_values is not None else None
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
    parser.add_argument('--add_hiding_square', action='store_true',
                       help='Add a square mask with 0 values above the oval to hide it')
    parser.add_argument('--square_size', nargs=3, type=int, default=None,
                       help='Size of hiding square (depth height width) (default: auto-calculated)')
    parser.add_argument('--square_offset', type=int, default=0,
                       help='Z-offset for square position relative to oval center (negative = above) (default: -15)')
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

    # Adjust output filename based on whether hiding square is added
    if args.add_hiding_square:
        output_path = output_dir / f"{args.patient_id}_lesion_hidden.nii.gz"
    else:
        output_path = output_dir / f"{args.patient_id}_lesion.nii.gz"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Processing patient {args.patient_id}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Lesion center (z,y,x): {args.center}")
    print(f"Semi-axes (a,b,c): {args.semi_axes}")
    if args.add_hiding_square:
        print(f"Adding hiding square: True")
        print(f"Square size: {args.square_size if args.square_size else 'auto-calculated'}")
        print(f"Square offset: {args.square_offset}")

    # Add the lesion
    try:
        stats = add_oval_lesion(
            str(input_path),
            str(output_path),
            tuple(args.center),
            tuple(args.semi_axes),
            args.lesion_value,
            add_hiding_square=args.add_hiding_square,
            square_size=tuple(args.square_size) if args.square_size else None,
            square_offset=args.square_offset
        )

        print("\n=== Lesion Creation Results ===")
        print(f"✓ Successfully created lesion for patient {args.patient_id}")
        print(f"  Center coordinates: {stats['center']}")
        print(f"  Semi-axes: {stats['semi_axes']}")
        print(f"  Lesion value: {stats['lesion_value']:.4f}")
        print(f"  Number of oval pixels: {stats['num_oval_pixels']}")
        print(f"  Original mean in oval area: {stats['original_mean_in_oval']:.4f}")
        print(f"  Original std in oval area: {stats['original_std_in_oval']:.4f}")

        if stats['hiding_square_added']:
            print(f"  Hiding square added: True")
            print(f"  Square size: {stats['square_size']}")
            print(f"  Square offset: {stats['square_offset']}")
            print(f"  Number of square pixels: {stats['num_square_pixels']}")
            print(f"  Original mean in square area: {stats['original_mean_in_square']:.4f}")
            print(f"  Original std in square area: {stats['original_std_in_square']:.4f}")

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
            f.write(f"Number of oval pixels: {stats['num_oval_pixels']}\n")
            f.write(f"Oval volume (voxels): {stats['oval_volume_voxels']}\n")
            f.write(f"Original mean in oval area: {stats['original_mean_in_oval']:.4f}\n")
            f.write(f"Original std in oval area: {stats['original_std_in_oval']:.4f}\n")

            if stats['hiding_square_added']:
                f.write(f"Hiding square added: True\n")
                f.write(f"Square size: {stats['square_size']}\n")
                f.write(f"Square offset: {stats['square_offset']}\n")
                f.write(f"Number of square pixels: {stats['num_square_pixels']}\n")
                f.write(f"Original mean in square area: {stats['original_mean_in_square']:.4f}\n")
                f.write(f"Original std in square area: {stats['original_std_in_square']:.4f}\n")
            else:
                f.write(f"Hiding square added: False\n")

            f.write(f"Image shape: {stats['image_shape']}\n")
            f.write(f"Input file: {input_path}\n")
            f.write(f"Output file: {output_path}\n")

        print(f"  Statistics saved to: {stats_file}")

    except Exception as e:
        print(f"Error creating lesion: {e}")
        return


if __name__ == "__main__":
    main()
