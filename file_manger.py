import csv
import json
import os
import shutil

from graph import plot_best_fitness_per_generation



def move_and_rename(OLD_DIR, SAVE_DIR):
    """Move and rename the save directory, then record high scores in CSV."""
    try:
        # Setup directories
        _ensure_directory_exists(OLD_DIR)
        
        # Move and rename operations
        if not _move_save_directory(OLD_DIR, SAVE_DIR):
            return  # Exit if source directory doesn't exist
        
        file_count = _get_file_count(OLD_DIR)
        new_name = _rename_moved_directory(file_count,OLD_DIR, SAVE_DIR)
        
        # Data processing
        plot_best_fitness_per_generation(new_name)
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


# Helper functions
def _ensure_directory_exists(dir_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def _move_save_directory(OLD_DIR, SAVE_DIR):
    """Move SAVE_DIR to OLD_DIR. Returns False if source doesn't exist."""
    if os.path.exists(SAVE_DIR):
        shutil.move(SAVE_DIR, OLD_DIR)
        print(f"Moved {SAVE_DIR} to {OLD_DIR}")
        return True
    print(f"Source directory {SAVE_DIR} does not exist")
    return False


def _get_file_count(directory):
    """Count files in the specified directory."""
    return len(os.listdir(directory))


def _rename_moved_directory(file_count,OLD_DIR, SAVE_DIR):
    """Rename the moved directory and return new path."""
    moved_path = os.path.join(OLD_DIR, os.path.basename(SAVE_DIR))
    new_name = os.path.join(OLD_DIR, f"dino_saves_{file_count}")
    os.rename(moved_path, new_name)
    print(f"Renamed {moved_path} to {new_name}")
    return new_name
