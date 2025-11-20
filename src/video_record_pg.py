import os
import sys
import time

import cv2


def get_next_filename(folder, base_name, extension=".avi"):
    """
    Generates the next filename in the sequence (e.g., video_1.avi, video_2.avi).
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        return os.path.join(folder, f"{base_name}_1{extension}")

    files = os.listdir(folder)
    max_num = 0
    for file in files:
        if file.startswith(base_name) and file.endswith(extension):
            try:
                # Extract number between base_name_ and .extension
                # Example: video_1.avi -> 1
                part = file.replace(base_name + "_", "").replace(extension, "")
                num = int(part)
                if num > max_num:
                    max_num = num
            except ValueError:
                continue

    return os.path.join(folder, f"{base_name}_{max_num + 1}{extension}")


def record_video():
    # Configuration
    VIDEO_FOLDER = "video"

    print("--- Video Recording Utility ---")

    # 0. Select Filename Prefix
    base_name_input = input("Enter filename prefix (default: 'recording'): ").strip()
    BASE_FILENAME = base_name_input if base_name_input else "recording"

    # 1. Select Resolution
    print("\nSelect Resolution:")
    print("1. 640x480 (Standard)")
    print("2. 1280x720 (HD)")
    print("3. 1920x1080 (Full HD)")
    print("4. Custom")

    res_choice = input("Enter choice (1-4): ").strip()

    width, height = 640, 480  # Default

    if res_choice == "1":
        width, height = 640, 480
    elif res_choice == "2":
        width, height = 1280, 720
    elif res_choice == "3":
        width, height = 1920, 1080
    elif res_choice == "4":
        try:
            width = int(input("Enter width: "))
            height = int(input("Enter height: "))
        except ValueError:
            print("Invalid input. Using default 640x480.")
    else:
        print("Invalid selection. Using default 640x480.")

    # 2. Select Duration
    try:
        duration = float(input("\nEnter recording duration in seconds: "))
    except ValueError:
        print("Invalid input. Defaulting to 10 seconds.")
        duration = 10.0

    # 3. Select File Format
    print("\nSelect File Format:")
    print("1. .avi (XVID)")
    print("2. .mp4 (mp4v)")

    fmt_choice = input("Enter choice (1-2): ").strip()
    if fmt_choice == "2":
        extension = ".mp4"
        codec_code = "mp4v"
    else:
        extension = ".avi"
        codec_code = "XVID"

    # Setup Camera
    cap = cv2.VideoCapture(0)

    # Set Resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Get actual resolution set by camera (it might not support the requested one)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera initialized at {actual_width}x{actual_height}")

    # Setup Video Writer
    output_path = get_next_filename(VIDEO_FOLDER, BASE_FILENAME, extension)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec_code)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (actual_width, actual_height))

    print(f"Recording started... will stop automatically after {duration} seconds.")
    print(f"Saving to: {output_path}")
    print("Press 'q' to stop early.")

    start_time = time.time()

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("\nError: Can't receive frame (stream end?). Exiting ...")
            break

        # Write the frame
        out.write(frame)

        # Display the resulting frame
        cv2.imshow("Recording Preview", frame)

        # Progress Bar
        elapsed = time.time() - start_time
        percent = min(100, int((elapsed / duration) * 100))
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = "=" * filled_length + "-" * (bar_length - filled_length)
        sys.stdout.write(f"\rProgress: [{bar}] {percent}%")
        sys.stdout.flush()

        if cv2.waitKey(1) == ord("q"):
            print("\nRecording stopped by user.")
            break

    # Clear progress bar line
    sys.stdout.write("\n")

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nRecording finished.")
    print(f"File saved: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    record_video()
