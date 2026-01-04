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

try:
    from utils.gif_generator import generate_gif
except ImportError:
    print("Error: utils.gif_generator not found.")
    sys.exit(1)


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

    # Get list of files to process
    files_to_process = []
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if not os.path.isabs(json_path):
            json_path = os.path.abspath(json_path)
        files_to_process = [json_path]
    else:
        # List available JSON files
        json_files = []
        if os.path.exists(keypoint_folder):
            json_files = sorted(
                [f for f in os.listdir(keypoint_folder) if f.endswith(".json")]
            )
            if json_files:
                print(f"\nAvailable keypoint files in {keypoint_folder}:")
                for i, f in enumerate(json_files, 1):
                    print(f"  {i}. {f}")
                print(f"  {len(json_files) + 1}. ALL FILES")
                print()

        json_path_input = input("Enter JSON file path (or number from list): ").strip()

        # Check if user entered a number
        if json_path_input.isdigit():
            idx = int(json_path_input) - 1
            if 0 <= idx < len(json_files):
                files_to_process = [os.path.join(keypoint_folder, json_files[idx])]
            elif idx == len(json_files):
                files_to_process = [
                    os.path.join(keypoint_folder, f) for f in json_files
                ]
            else:
                print("Invalid selection.")
                return
        else:
            # Handle manual path entry
            if not json_path_input:
                print("No input provided.")
                return

            if os.path.isabs(json_path_input):
                json_path = json_path_input
            elif os.path.exists(os.path.join(keypoint_folder, json_path_input)):
                json_path = os.path.join(keypoint_folder, json_path_input)
            else:
                json_path = os.path.abspath(json_path_input)
            files_to_process = [json_path]

    # Validate all files exist
    valid_files = []
    for f in files_to_process:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            print(f"Error: File not found: {f}")

    if not valid_files:
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

    # Process each file
    for json_path in valid_files:
        print()
        print("-" * 40)
        print(f"Processing: {os.path.basename(json_path)}")

        output = generate_gif(
            json_path=json_path,
            output_path=output_path,
            fps=fps,
            panel_size=(320, 240),
        )
        print(f"GIF generated: {output}")

    print("\n" + "=" * 60)
    print(f"  Processing complete! Total files: {len(valid_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
