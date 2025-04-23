import os
import json
from pathlib import Path
import argparse
from tqdm import tqdm

def find_and_delete_unknown_chunks_files(analysis_dir):
    """
    Find and delete chunks.json files that contain only 'Unknown' categories.
    
    Args:
        analysis_dir: Path to the analysis directory
    """
    analysis_path = Path(analysis_dir)
    
    # Count variables for reporting
    total_files = 0
    deleted_files = 0
    
    # Find all problem directories
    problem_dirs = [d for d in analysis_path.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    
    for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
        # Find all seed directories
        seed_dirs = [d for d in problem_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        for seed_dir in seed_dirs:
            chunks_file = seed_dir / "chunks.json"
            
            # Check if chunks.json exists
            if chunks_file.exists():
                total_files += 1
                
                try:
                    # Load the chunks data
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    
                    # Check if all chunks have 'Unknown' category
                    all_unknown = all(chunk.get("category", "Unknown") == "Unknown" for chunk in chunks_data)
                    
                    if all_unknown:
                        # Delete the file
                        chunks_file.unlink()
                        deleted_files += 1
                        print(f"Deleted: {chunks_file}")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON in {chunks_file}")
                except Exception as e:
                    print(f"Error processing {chunks_file}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"Total chunks.json files found: {total_files}")
    print(f"Files deleted (all Unknown): {deleted_files}")
    print(f"Remaining files: {total_files - deleted_files}")

def main():
    parser = argparse.ArgumentParser(description="Delete chunks.json files with only 'Unknown' categories")
    parser.add_argument("--analysis_dir", type=str, default="../analysis", help="Path to the analysis directory")
    parser.add_argument("--dry_run", action="store_true", help="Don't actually delete files, just report what would be deleted")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE: No files will be deleted")
        
    # Backup function to use during dry run
    original_unlink = Path.unlink
    
    if args.dry_run:
        # Override the unlink method to just print instead of deleting
        def dry_run_unlink(self, *args, **kwargs):
            print(f"Would delete: {self}")
        
        Path.unlink = dry_run_unlink
    
    try:
        find_and_delete_unknown_chunks_files(args.analysis_dir)
    finally:
        # Restore original unlink method if we modified it
        if args.dry_run:
            Path.unlink = original_unlink

if __name__ == "__main__":
    main()