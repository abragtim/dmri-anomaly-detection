#!/usr/bin/env python3
"""
Script to remove all files that don't have .zip extension from the hcp_data directory.
This will remove .md5 checksum files and any other non-zip files.
"""

import os
import sys
from pathlib import Path


def filter_data(data_dir="hcp_data", dry_run=False):
    """
    Remove all files that don't have .zip extension from the specified directory.

    Args:
        data_dir (str): Directory path to filter files from
        dry_run (bool): If True, only show what would be deleted without actually deleting
    """
    # Get the script directory and construct the data directory path
    script_dir = Path(__file__).parent
    data_path = script_dir / data_dir

    if not data_path.exists():
        print(f"Error: Directory '{data_path}' does not exist!")
        return False

    if not data_path.is_dir():
        print(f"Error: '{data_path}' is not a directory!")
        return False

    # Find all files that don't have .zip extension
    files_to_remove = []
    for file_path in data_path.iterdir():
        if file_path.is_file() and not file_path.name.endswith('.zip'):
            files_to_remove.append(file_path)

    if not files_to_remove:
        print("No non-zip files found to remove.")
        return True

    print(f"Found {len(files_to_remove)} non-zip files to remove:")
    for file_path in files_to_remove:
        print(f"  - {file_path.name}")

    if dry_run:
        print("\nDry run mode - no files were actually deleted.")
        return True

    # Ask for confirmation
    response = input(f"\nAre you sure you want to delete these {len(files_to_remove)} files? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Operation cancelled.")
        return False

    # Remove the files
    removed_count = 0
    failed_count = 0

    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"Removed: {file_path.name}")
            removed_count += 1
        except OSError as e:
            print(f"Failed to remove {file_path.name}: {e}")
            failed_count += 1

    print(f"\nOperation completed:")
    print(f"  - Successfully removed: {removed_count} files")
    if failed_count > 0:
        print(f"  - Failed to remove: {failed_count} files")

    return failed_count == 0


def main():
    """Main function to handle command line arguments and run the filter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove all non-zip files from the hcp_data directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting files"
    )
    parser.add_argument(
        "--data-dir",
        default="hcp_data",
        help="Directory to filter files from (default: hcp_data)"
    )

    args = parser.parse_args()

    print("HCP Data Filter Script")
    print("=" * 50)

    if args.dry_run:
        print("Running in DRY RUN mode - no files will be deleted")
        print("=" * 50)

    success = filter_data(args.data_dir, args.dry_run)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
