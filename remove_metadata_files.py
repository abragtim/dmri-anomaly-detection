"""
These ._ files are created by macOS when files are copied to non-HFS+ file systems
(like external drives, network shares, etc.) and contain extended attributes and metadata.
"""

import os
import argparse
import sys


def find_metadata_files(directory):
    """Find all ._ metadata files in directory and subdirectories."""
    metadata_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('._'):
                file_path = os.path.join(root, file)
                metadata_files.append(file_path)
    
    return metadata_files


def remove_metadata_files(directory, dry_run=False):
    """Remove all ._ metadata files from directory and subdirectories."""
    metadata_files = find_metadata_files(directory)
    
    if not metadata_files:
        print("No ._ metadata files found.")
        return 0
    
    print(f"Found {len(metadata_files)} ._ metadata files:")
    
    for file_path in metadata_files:
        relative_path = os.path.relpath(file_path, directory)
        
        if dry_run:
            print(f"[DRY RUN] Would remove: {relative_path}")
        else:
            try:
                os.remove(file_path)
                print(f"Removed: {relative_path}")
            except OSError as e:
                print(f"Error removing {relative_path}: {e}", file=sys.stderr)
    
    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(metadata_files)} files.")
        print("Run without --dry-run to actually remove the files.")
    else:
        print(f"\nRemoved {len(metadata_files)} ._ metadata files.")
    
    return len(metadata_files)


def main():
    parser = argparse.ArgumentParser(
        description="Remove macOS metadata files (._ files) from current directory and all subdirectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_metadata_files.py --dry-run    # Preview what would be removed
  python remove_metadata_files.py              # Actually remove the files
  
Note: This script removes ._ files created by macOS when copying files to 
non-HFS+ file systems. These files contain extended attributes and metadata.
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what files would be removed without actually removing them'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        default='.',
        help='Directory to clean (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Get absolute path of the directory
    directory = os.path.abspath(args.directory)
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Scanning directory: {directory}")
    
    if args.dry_run:
        print("[DRY RUN MODE] - No files will be actually removed")
    
    print("-" * 50)
    
    try:
        count = remove_metadata_files(directory, args.dry_run)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()