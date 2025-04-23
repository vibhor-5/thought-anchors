#!/usr/bin/env python3
"""
Script to clean up the analysis folder by removing .npy and .png files
while preserving chunks.json files.
"""

import os
from pathlib import Path
import shutil
from tqdm import tqdm

def cleanup_analysis_folder(analysis_dir: Path):
    """
    Remove all .npy and .png files from the analysis directory and its subdirectories,
    while keeping chunks.json files.
    
    Args:
        analysis_dir: Path to the analysis directory
    """
    # Check if the directory exists
    if not analysis_dir.exists():
        print(f"Analysis directory {analysis_dir} does not exist.")
        return
    
    # Count files to be deleted for progress tracking
    total_files = 0
    for root, _, files in os.walk(analysis_dir):
        for file in files:
            if file.endswith(('.npy', '.png')):
                total_files += 1
    
    print(f"Found {total_files} .npy and .png files to delete.")
    
    # Delete files with progress bar
    deleted_count = 0
    with tqdm(total=total_files, desc="Deleting files") as pbar:
        for root, _, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith(('.npy', '.png')):
                    file_path = Path(root) / file
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
    
    print(f"Successfully deleted {deleted_count} files.")
    print(f"Kept all chunks.json files.")

if __name__ == "__main__":
    # Set the path to the analysis directory
    analysis_dir = Path("../analysis")
    
    # Confirm before proceeding
    print(f"This will delete all .npy and .png files in {analysis_dir} and its subdirectories.")
    confirmation = input("Do you want to proceed? (y/n): ")
    
    if confirmation.lower() == 'y':
        cleanup_analysis_folder(analysis_dir)
    else:
        print("Operation cancelled.")