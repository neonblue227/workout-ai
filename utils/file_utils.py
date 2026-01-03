"""
File utility functions.

Provides helpers for file naming and directory management.
"""

import os


def get_next_filename(folder, base_name, extension=".mp4"):
    """
    Generate next filename in sequence.

    Creates directory if it doesn't exist. Finds existing files with the
    same base name and returns the next sequential number.

    Args:
        folder: Directory path to save files
        base_name: Base filename prefix (e.g., "recording")
        extension: File extension including dot (default: ".mp4")

    Returns:
        str: Full path to next file (e.g., "/path/recording_3.mp4")
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        return os.path.join(folder, f"{base_name}_1{extension}")

    files = os.listdir(folder)
    max_num = 0
    for file in files:
        if file.startswith(base_name) and file.endswith(extension):
            try:
                part = file.replace(base_name + "_", "").replace(extension, "")
                num = int(part)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return os.path.join(folder, f"{base_name}_{max_num + 1}{extension}")
