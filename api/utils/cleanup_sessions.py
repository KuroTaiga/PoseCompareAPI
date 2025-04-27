#!/usr/bin/env python
"""
Script to clean up expired sessions.
Can be run manually or as a scheduled task.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import FileManager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_manager import FileManager

def cleanup_sessions(storage_dir="storage", max_age_hours=5, verbose=False):
    """
    Clean up expired sessions
    
    Args:
        storage_dir (str): Storage directory path
        max_age_hours (int): Maximum age of sessions in hours
        verbose (bool): Print detailed information
        
    Returns:
        int: Number of sessions deleted
    """
    if verbose:
        print(f"Starting session cleanup at {datetime.now().isoformat()}")
        print(f"Looking for sessions older than {max_age_hours} hours in {storage_dir}")
    
    file_manager = FileManager(storage_dir)
    deleted_count = file_manager.cleanup_expired_sessions(max_age_hours)
    
    if verbose:
        print(f"Cleanup complete. Deleted {deleted_count} expired sessions.")
    
    return deleted_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up expired sessions")
    parser.add_argument(
        "--storage", 
        default="storage",
        help="Storage directory path (default: storage)"
    )
    parser.add_argument(
        "--max-age", 
        type=int, 
        default=5,
        help="Maximum age of sessions in hours (default: 5)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    cleanup_sessions(args.storage, args.max_age, args.verbose)