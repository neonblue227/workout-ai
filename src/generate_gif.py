"""
GIF Generator CLI Script.

Generates animated GIFs from JSON keypoint data with body, hands,
and face visualizations in a 2x2 grid layout.
"""

import os
import sys

# Path hack for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.gif_generator import generate_gif  # noqa: E402


def main():
    """Main entry point for GIF generation."""
    print("=" * 60)
    print("  GIF Generator from JSON Keypoints")
    print("  Body + Hands + Face in 2x2 Grid Layout")
    print("=" * 60)

    # Default paths
    keypoint_folder = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "keypoint")
    )

    # Get input file
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if not os.path.isabs(json_path):
            json_path = os.path.abspath(json_path)
    else:
        # List available JSON files
        if os.path.exists(keypoint_folder):
            json_files = [f for f in os.listdir(keypoint_folder) if f.endswith(".json")]
            if json_files:
                print(f"\nAvailable keypoint files in {keypoint_folder}:")
                for i, f in enumerate(json_files, 1):
                    print(f"  {i}. {f}")
                print()

        json_path = input("Enter JSON file path (or number from list): ").strip()

        # Check if user entered a number
        if json_path.isdigit():
            idx = int(json_path) - 1
            if 0 <= idx < len(json_files):
                json_path = os.path.join(keypoint_folder, json_files[idx])
            else:
                print("Invalid selection.")
                return
        elif not os.path.isabs(json_path):
            # Try relative to keypoint folder first
            if os.path.exists(os.path.join(keypoint_folder, json_path)):
                json_path = os.path.join(keypoint_folder, json_path)
            else:
                json_path = os.path.abspath(json_path)

    # Validate input
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return

    # Get optional parameters
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

    fps = None
    if len(sys.argv) > 3:
        try:
            fps = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid FPS value '{sys.argv[3]}', using default")

    # Generate GIF
    print()
    output = generate_gif(
        json_path=json_path,
        output_path=output_path,
        fps=fps,
        panel_size=(320, 240),
    )
    print()
    print("=" * 60)
    print("  GIF generated successfully!")
    print(f"  Output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
