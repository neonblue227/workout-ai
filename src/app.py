"""
MoveUp-AI GUI Application.

A modern Tkinter interface for video recording with MediaPipe
keypoint overlay and GIF generation.
"""

# Standard Library
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from queue import Empty, Queue
from tkinter import (
    Button,
    Entry,
    Frame,
    Label,
    PanedWindow,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
    ttk,
)

# Third-Party
import cv2
from PIL import Image, ImageTk

# Path hack for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try to import MediaPipe and local utils
try:
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_holistic = None

try:
    from utils import (
        KeypointRecorder,
        draw_face_mesh,
        draw_hand_landmarks,
        draw_pose_landmarks,
        extract_all_keypoints,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


# =============================================================================
# Theme & Styling
# =============================================================================


class Theme:
    """Windows 10 light theme colors and fonts."""

    # Colors - Windows 10 Light Theme
    BG_WINDOW = "#f0f0f0"  # Standard window background
    BG_CONTROL = "#ffffff"  # Control/input background
    BG_PANEL = "#e5e5e5"  # Panel/section background
    BORDER = "#adadad"  # Border color
    ACCENT = "#0078d4"  # Windows 10 blue accent
    ACCENT_HOVER = "#106ebe"  # Accent hover state
    TEXT_PRIMARY = "#1a1a1a"  # Primary text (near black)
    TEXT_SECONDARY = "#5c5c5c"  # Secondary/muted text
    SUCCESS = "#107c10"  # Green for success states
    WARNING = "#ca5010"  # Orange for warnings
    DISABLED = "#a0a0a0"  # Disabled state

    # Aliases for backwards compatibility
    BG_DARK = BG_WINDOW
    BG_MEDIUM = BG_PANEL
    BG_LIGHT = BG_CONTROL

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_TITLE = (FONT_FAMILY, 14, "bold")
    FONT_HEADING = (FONT_FAMILY, 11, "bold")
    FONT_NORMAL = (FONT_FAMILY, 10)
    FONT_SMALL = (FONT_FAMILY, 9)
    FONT_MONO = ("Consolas", 9)


# =============================================================================
# Placeholder Functions (Replace with actual imports when available)
# =============================================================================


def get_mediapipe_landmarks(frame, holistic):
    """
    Get MediaPipe Holistic landmarks from a frame.

    This is the main integration point - replace the processing logic
    as needed for your specific use case.

    Args:
        frame: BGR OpenCV frame
        holistic: MediaPipe Holistic instance

    Returns:
        MediaPipe results object
    """
    if holistic is None:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = holistic.process(rgb_frame)
    rgb_frame.flags.writeable = True
    return results


def draw_landmarks_on_frame(frame, results, visibility_flags):
    """
    Draw MediaPipe landmarks on the frame based on visibility flags.

    Args:
        frame: BGR OpenCV frame to draw on
        results: MediaPipe Holistic results
        visibility_flags: dict with keys 'body', 'face', 'left_hand', 'right_hand'

    Returns:
        Frame with landmarks drawn
    """
    if results is None:
        return frame

    overlay = frame.copy()

    # Use actual utils if available, otherwise fallback to basic drawing
    if UTILS_AVAILABLE:
        if visibility_flags.get("body", True) and results.pose_landmarks:
            draw_pose_landmarks(
                overlay,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS if MEDIAPIPE_AVAILABLE else None,
            )

        if visibility_flags.get("face", True) and results.face_landmarks:
            draw_face_mesh(overlay, results.face_landmarks)

        if visibility_flags.get("left_hand", True) and results.left_hand_landmarks:
            mp_hands = mp.solutions.hands if MEDIAPIPE_AVAILABLE else None
            draw_hand_landmarks(
                overlay,
                results.left_hand_landmarks,
                "Left",
                mp_hands.HAND_CONNECTIONS if mp_hands else None,
            )

        if visibility_flags.get("right_hand", True) and results.right_hand_landmarks:
            mp_hands = mp.solutions.hands if MEDIAPIPE_AVAILABLE else None
            draw_hand_landmarks(
                overlay,
                results.right_hand_landmarks,
                "Right",
                mp_hands.HAND_CONNECTIONS if mp_hands else None,
            )
    else:
        # Basic fallback drawing using MediaPipe's built-in drawing
        if MEDIAPIPE_AVAILABLE:
            mp_draw = mp.solutions.drawing_utils
            mp_draw_styles = mp.solutions.drawing_styles

            if visibility_flags.get("body", True) and results.pose_landmarks:
                mp_draw.draw_landmarks(
                    overlay,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_draw_styles.get_default_pose_landmarks_style(),
                )

            if visibility_flags.get("face", True) and results.face_landmarks:
                mp_draw.draw_landmarks(
                    overlay,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw_styles.get_default_face_mesh_tesselation_style(),
                )

            if visibility_flags.get("left_hand", True) and results.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    overlay,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style(),
                )

            if (
                visibility_flags.get("right_hand", True)
                and results.right_hand_landmarks
            ):
                mp_draw.draw_landmarks(
                    overlay,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_draw_styles.get_default_hand_landmarks_style(),
                    mp_draw_styles.get_default_hand_connections_style(),
                )

    return overlay


# =============================================================================
# Camera Thread
# =============================================================================


class CameraThread(threading.Thread):
    """Background thread for webcam capture and MediaPipe processing."""

    def __init__(self, camera_id, frame_queue, log_queue, visibility_flags):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.visibility_flags = visibility_flags
        self.running = False
        self.recording = False
        self.record_data = None
        self.fps = 0
        self.cap = None
        self.holistic = None
        self._reset_frame_count = False

    def run(self):
        """Main camera loop."""
        self.running = True

        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.log_queue.put(f"Error: Could not open camera {self.camera_id}")
            self.running = False
            return

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log_queue.put(f"Camera initialized: {actual_width}x{actual_height}")

        # Initialize MediaPipe Holistic
        if MEDIAPIPE_AVAILABLE:
            self.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,  # Balance between speed and accuracy
                smooth_landmarks=True,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.log_queue.put("MediaPipe Holistic initialized")
        else:
            self.log_queue.put("Warning: MediaPipe not available")

        frame_count = 0
        fps_start = time.time()
        fps_frames = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Get MediaPipe results
            results = get_mediapipe_landmarks(frame, self.holistic)

            # Draw landmarks based on visibility flags
            display_frame = draw_landmarks_on_frame(
                frame, results, self.visibility_flags
            )

            # If recording, process and store data
            if self.recording and self.record_data is not None:
                # Reset frame count for new recording session
                if self._reset_frame_count:
                    frame_count = 0
                    self._reset_frame_count = False

                elapsed = time.time() - self.record_data["start_time"]

                if UTILS_AVAILABLE and results is not None:
                    # Extract keypoints
                    keypoints = extract_all_keypoints(results, frame.shape)
                    self.record_data["recorder"].add_frame(
                        frame_count, elapsed, keypoints
                    )

                    # Build log message
                    detected = []
                    if results.pose_landmarks:
                        detected.append("Pose")
                    if results.face_landmarks:
                        detected.append("Face")
                    if results.left_hand_landmarks:
                        detected.append("L-Hand")
                    if results.right_hand_landmarks:
                        detected.append("R-Hand")

                    self.log_queue.put(
                        f"Frame {frame_count}: {', '.join(detected) if detected else 'No detection'}"
                    )

                # Write frame to video
                if self.record_data.get("video_writer"):
                    self.record_data["video_writer"].write(display_frame)

                self.record_data["frame_count"] = frame_count
                self.record_data["elapsed"] = elapsed
                frame_count += 1

            # Calculate FPS
            fps_frames += 1
            if time.time() - fps_start >= 1.0:
                self.fps = fps_frames
                fps_frames = 0
                fps_start = time.time()

            # Convert to RGB for Tkinter (send original size, scaling happens in UI)
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Put frame in queue (drop old frames if queue is full)
            try:
                while not self.frame_queue.empty():
                    self.frame_queue.get_nowait()
            except Empty:
                pass
            self.frame_queue.put((rgb_frame, self.fps))

        # Cleanup
        if self.holistic:
            self.holistic.close()
        if self.cap:
            self.cap.release()

    def start_recording(self, filename, duration, save_dir):
        """Start recording video and keypoints."""
        if self.cap is None:
            return False

        video_dir = os.path.join(save_dir, "video")
        keypoint_dir = os.path.join(save_dir, "keypoint")
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(keypoint_dir, exist_ok=True)

        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{filename}_{timestamp}.mp4"
        video_path = os.path.join(video_dir, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

        # Create keypoint recorder
        keypoint_recorder = None
        if UTILS_AVAILABLE:
            keypoint_recorder = KeypointRecorder(
                base_filename=f"{filename}_{timestamp}",
                resolution=(width, height),
                fps=30.0,
            )

        self.record_data = {
            "start_time": time.time(),
            "duration": duration,
            "video_writer": video_writer,
            "video_path": video_path,
            "recorder": keypoint_recorder,
            "keypoint_dir": keypoint_dir,
            "frame_count": 0,
            "elapsed": 0,
        }
        self.recording = True
        self._reset_frame_count = True  # Signal to reset frame count in run loop
        self.log_queue.put(f"Recording started: {video_filename}")
        return True

    def stop_recording(self):
        """Stop recording and save files."""
        if not self.recording:
            return None

        self.recording = False

        if self.record_data:
            # Close video writer
            if self.record_data.get("video_writer"):
                self.record_data["video_writer"].release()

            # Save keypoints
            keypoint_path = None
            if self.record_data.get("recorder"):
                keypoint_path = self.record_data["recorder"].save(
                    self.record_data["keypoint_dir"]
                )

            result = {
                "video_path": self.record_data.get("video_path"),
                "keypoint_path": keypoint_path,
                "frame_count": self.record_data.get("frame_count", 0),
            }

            self.log_queue.put(f"Recording saved: {result['frame_count']} frames")
            self.record_data = None
            return result

        return None

    def stop(self):
        """Stop the camera thread."""
        self.running = False


# =============================================================================
# Main Application
# =============================================================================


class MoveUpApp:
    """Main MoveUp-AI GUI Application."""

    def __init__(self):
        self.root = Tk()
        self.root.title("MoveUp-AI - Posture Detection")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg=Theme.BG_DARK)

        # Configure styles
        self.setup_styles()

        # Variables
        self.filename_var = StringVar(value="recording")
        self.duration_var = StringVar(value="30")
        self.save_dir_var = StringVar(value=os.path.join(project_root, "data"))
        self.camera_var = StringVar(value="0")
        self.fps_var = StringVar(value="FPS: --")
        self.status_var = StringVar(value="Ready")
        self.progress_var = StringVar(value="0 / 0 sec")

        # Visibility toggles
        self.visibility_flags = {
            "body": True,
            "face": True,
            "left_hand": True,
            "right_hand": True,
        }
        self.toggle_buttons = {}

        # Threading
        self.frame_queue = Queue(maxsize=3)
        self.log_queue = Queue()
        self.camera_thread = None
        self.recording_active = False
        self.recording_start_time = None
        self.recording_duration = 0

        # GIF display
        self.gif_frames = []
        self.gif_index = 0
        self.last_keypoint_path = None

        # Build UI
        self.build_ui()

        # Start update loops
        self.update_camera_display()
        self.update_log_display()
        self.update_recording_progress()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_styles(self):
        """Configure ttk styles for Windows 10 look."""
        style = ttk.Style()
        style.theme_use("vista")  # Use vista theme for Windows native look

        # Configure combobox to match light theme
        style.configure(
            "TCombobox",
            fieldbackground=Theme.BG_CONTROL,
            background=Theme.BG_CONTROL,
            foreground=Theme.TEXT_PRIMARY,
        )

    def build_ui(self):
        """Build the main UI layout."""
        # Main container with padding
        main_frame = Frame(self.root, bg=Theme.BG_DARK, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        # Title bar
        title_frame = Frame(main_frame, bg=Theme.BG_DARK)
        title_frame.pack(fill="x", pady=(0, 15))

        title_label = Label(
            title_frame,
            text="MoveUp-AI",
            font=Theme.FONT_TITLE,
            bg=Theme.BG_DARK,
            fg=Theme.ACCENT,
        )
        title_label.pack(side="left")

        status_label = Label(
            title_frame,
            textvariable=self.status_var,
            font=Theme.FONT_SMALL,
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY,
        )
        status_label.pack(side="right")

        # Content area with resizable two columns using PanedWindow
        self.paned_window = PanedWindow(
            main_frame,
            orient="horizontal",
            bg=Theme.BORDER,
            sashwidth=6,
            sashrelief="raised",
        )
        self.paned_window.pack(fill="both", expand=True)

        # Left column (camera and controls)
        left_column = Frame(self.paned_window, bg=Theme.BG_DARK)
        self.paned_window.add(left_column, minsize=400, stretch="always")

        self.build_left_column(left_column)

        # Right column (data management)
        right_column = Frame(self.paned_window, bg=Theme.BG_MEDIUM)
        self.paned_window.add(right_column, minsize=300, stretch="always")

        self.build_right_column(right_column)

    def build_left_column(self, parent):
        """Build the left column with camera and controls."""
        # Settings area
        settings_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        # File name input
        name_frame = Frame(settings_frame, bg=Theme.BG_MEDIUM)
        name_frame.pack(fill="x", pady=5)

        Label(
            name_frame,
            text="File Name:",
            font=Theme.FONT_NORMAL,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
            width=12,
            anchor="w",
        ).pack(side="left")

        Entry(
            name_frame,
            textvariable=self.filename_var,
            font=Theme.FONT_NORMAL,
            bg=Theme.BG_CONTROL,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            relief="sunken",
            bd=1,
            width=25,
        ).pack(side="left", padx=5)

        # Duration input
        Label(
            name_frame,
            text="Duration (s):",
            font=Theme.FONT_NORMAL,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
            width=12,
            anchor="w",
        ).pack(side="left", padx=(20, 0))

        Entry(
            name_frame,
            textvariable=self.duration_var,
            font=Theme.FONT_NORMAL,
            bg=Theme.BG_CONTROL,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            relief="sunken",
            bd=1,
            width=8,
        ).pack(side="left", padx=5)

        # Camera display area
        camera_frame = Frame(parent, bg=Theme.BG_LIGHT, padx=3, pady=3)
        camera_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.camera_label = Label(
            camera_frame,
            bg="#2d2d2d",
            text="Camera Feed\n\nInitializing...",
            font=Theme.FONT_HEADING,
            fg="#ffffff",
        )
        self.camera_label.pack(fill="both", expand=True)

        # Camera controls
        controls_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=10)
        controls_frame.pack(fill="x", pady=(0, 10))

        # Camera selection
        cam_select_frame = Frame(controls_frame, bg=Theme.BG_MEDIUM)
        cam_select_frame.pack(fill="x", pady=5)

        Label(
            cam_select_frame,
            text="Camera:",
            font=Theme.FONT_NORMAL,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
        ).pack(side="left")

        camera_combo = ttk.Combobox(
            cam_select_frame,
            textvariable=self.camera_var,
            values=["0", "1", "2", "3"],
            width=8,
            state="readonly",
        )
        camera_combo.pack(side="left", padx=10)
        camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # FPS display
        fps_label = Label(
            cam_select_frame,
            textvariable=self.fps_var,
            font=Theme.FONT_MONO,
            bg=Theme.BG_MEDIUM,
            fg=Theme.SUCCESS,
        )
        fps_label.pack(side="left", padx=20)

        # Progress display
        progress_label = Label(
            cam_select_frame,
            textvariable=self.progress_var,
            font=Theme.FONT_MONO,
            bg=Theme.BG_MEDIUM,
            fg=Theme.WARNING,
        )
        progress_label.pack(side="right")

        # Action buttons
        button_frame = Frame(parent, bg=Theme.BG_DARK)
        button_frame.pack(fill="x")

        self.start_btn = Button(
            button_frame,
            text="▶ Start Recording",
            font=Theme.FONT_HEADING,
            bg=Theme.ACCENT,
            fg="#ffffff",
            activebackground=Theme.ACCENT_HOVER,
            activeforeground="#ffffff",
            relief="raised",
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.start_recording,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = Button(
            button_frame,
            text="■ Stop",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_PANEL,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BORDER,
            activeforeground=Theme.TEXT_PRIMARY,
            relief="raised",
            padx=20,
            pady=8,
            cursor="hand2",
            state="disabled",
            command=self.stop_recording,
        )
        self.stop_btn.pack(side="left")

    def build_right_column(self, parent):
        """Build the right column with data management."""
        # Directory settings
        dir_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=15)
        dir_frame.pack(fill="x")

        Label(
            dir_frame,
            text="Save Directory",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")

        dir_input_frame = Frame(dir_frame, bg=Theme.BG_MEDIUM)
        dir_input_frame.pack(fill="x", pady=(5, 0))

        self.dir_entry = Entry(
            dir_input_frame,
            textvariable=self.save_dir_var,
            font=Theme.FONT_SMALL,
            bg=Theme.BG_CONTROL,
            fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            relief="sunken",
            bd=1,
        )
        self.dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        Button(
            dir_input_frame,
            text="Browse",
            font=Theme.FONT_SMALL,
            bg=Theme.BG_PANEL,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BORDER,
            relief="raised",
            padx=10,
            cursor="hand2",
            command=self.browse_directory,
        ).pack(side="right")

        # JSON Log display
        log_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=10)
        log_frame.pack(fill="both", expand=True)

        Label(
            log_frame,
            text="Raw JSON Log",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w")

        self.log_text = Text(
            log_frame,
            font=Theme.FONT_MONO,
            bg=Theme.BG_CONTROL,
            fg=Theme.TEXT_PRIMARY,
            relief="sunken",
            bd=1,
            height=10,
            wrap="word",
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True, pady=(5, 0))

        # Visualization toggles
        toggle_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=10)
        toggle_frame.pack(fill="x")

        Label(
            toggle_frame,
            text="Visualization Toggles",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
        ).pack(anchor="w", pady=(0, 10))

        toggles_grid = Frame(toggle_frame, bg=Theme.BG_MEDIUM)
        toggles_grid.pack(fill="x")

        toggle_labels = [
            ("body", "Body"),
            ("face", "Face"),
            ("left_hand", "L Hand"),
            ("right_hand", "R Hand"),
        ]

        for i, (key, label) in enumerate(toggle_labels):
            btn = Button(
                toggles_grid,
                text=label,
                font=Theme.FONT_NORMAL,
                bg=Theme.SUCCESS,
                fg="#ffffff",
                activebackground=Theme.ACCENT_HOVER,
                relief="raised",
                width=10,
                pady=4,
                cursor="hand2",
                command=lambda k=key: self.toggle_visibility(k),
            )
            btn.grid(row=i // 2, column=i % 2, padx=5, pady=5, sticky="ew")
            self.toggle_buttons[key] = btn

        toggles_grid.columnconfigure(0, weight=1)
        toggles_grid.columnconfigure(1, weight=1)

        # GIF Preview area
        gif_frame = Frame(parent, bg=Theme.BG_MEDIUM, padx=15, pady=10)
        gif_frame.pack(fill="x")

        gif_header = Frame(gif_frame, bg=Theme.BG_MEDIUM)
        gif_header.pack(fill="x")

        Label(
            gif_header,
            text="GIF Preview",
            font=Theme.FONT_HEADING,
            bg=Theme.BG_MEDIUM,
            fg=Theme.TEXT_PRIMARY,
        ).pack(side="left")

        Button(
            gif_header,
            text="Generate GIF",
            font=Theme.FONT_SMALL,
            bg=Theme.ACCENT,
            fg="#ffffff",
            activebackground=Theme.ACCENT_HOVER,
            relief="raised",
            padx=15,
            pady=3,
            cursor="hand2",
            command=self.generate_gif,
        ).pack(side="right")

        Button(
            gif_header,
            text="Run GIF CLI",
            font=Theme.FONT_SMALL,
            bg=Theme.BG_PANEL,
            fg=Theme.TEXT_PRIMARY,
            activebackground=Theme.BORDER,
            relief="raised",
            padx=15,
            pady=3,
            cursor="hand2",
            command=self.run_gif_cli,
        ).pack(side="right", padx=(0, 10))

        self.gif_label = Label(
            gif_frame,
            bg=Theme.BG_CONTROL,
            height=8,
            text="No GIF generated",
            font=Theme.FONT_SMALL,
            fg=Theme.TEXT_SECONDARY,
            relief="sunken",
            bd=1,
        )
        self.gif_label.pack(fill="x", pady=(10, 0))

    def toggle_visibility(self, key):
        """Toggle landmark visibility."""
        self.visibility_flags[key] = not self.visibility_flags[key]
        btn = self.toggle_buttons[key]

        if self.visibility_flags[key]:
            btn.configure(bg=Theme.SUCCESS, fg="#ffffff")
        else:
            btn.configure(bg=Theme.BG_PANEL, fg=Theme.TEXT_SECONDARY)

    def browse_directory(self):
        """Open directory browser dialog."""
        path = filedialog.askdirectory(
            initialdir=self.save_dir_var.get(),
            title="Select Save Directory",
        )
        if path:
            self.save_dir_var.set(path)

    def on_camera_change(self, event=None):
        """Handle camera selection change."""
        if self.camera_thread and self.camera_thread.running:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=2)

        self.start_camera()

    def start_camera(self):
        """Start the camera thread."""
        camera_id = int(self.camera_var.get())

        self.camera_thread = CameraThread(
            camera_id,
            self.frame_queue,
            self.log_queue,
            self.visibility_flags,
        )
        self.camera_thread.start()
        self.status_var.set(f"Camera {camera_id} active")

    def start_recording(self):
        """Start video and keypoint recording."""
        if self.camera_thread is None or not self.camera_thread.running:
            messagebox.showerror("Error", "Camera is not running")
            return

        try:
            duration = float(self.duration_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid duration value")
            return

        filename = self.filename_var.get().strip()
        if not filename:
            filename = "recording"

        save_dir = self.save_dir_var.get()

        success = self.camera_thread.start_recording(filename, duration, save_dir)

        if success:
            self.recording_active = True
            self.recording_start_time = time.time()
            self.recording_duration = duration

            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal", bg=Theme.ACCENT)
            self.status_var.set("Recording...")

    def stop_recording(self):
        """Stop recording."""
        if self.camera_thread:
            result = self.camera_thread.stop_recording()

            if result:
                self.last_keypoint_path = result.get("keypoint_path")
                self.add_log(f"Saved: {result['video_path']}")
                if result.get("keypoint_path"):
                    self.add_log(f"Keypoints: {result['keypoint_path']}")

        self.recording_active = False
        self.recording_start_time = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled", bg=Theme.BG_LIGHT)
        self.status_var.set("Ready")
        self.progress_var.set("0 / 0 sec")

    def update_recording_progress(self):
        """Update recording progress display."""
        if self.recording_active and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            self.progress_var.set(f"{elapsed:.1f} / {self.recording_duration:.0f} sec")

            # Auto-stop when duration reached
            if elapsed >= self.recording_duration:
                self.stop_recording()

        self.root.after(100, self.update_recording_progress)

    def update_camera_display(self):
        """Update camera display from queue, scaling to fit the label."""
        try:
            frame, fps = self.frame_queue.get_nowait()

            # Get the label's current size
            label_width = self.camera_label.winfo_width()
            label_height = self.camera_label.winfo_height()

            # Only resize if we have valid dimensions
            if label_width > 1 and label_height > 1:
                # Convert to PIL Image
                pil_image = Image.fromarray(frame)

                # Calculate scale to fit while maintaining aspect ratio
                img_width, img_height = pil_image.size
                scale = min(label_width / img_width, label_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                # Resize the image to fit the label
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
            else:
                pil_image = Image.fromarray(frame)

            photo = ImageTk.PhotoImage(pil_image)

            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo

            self.fps_var.set(f"FPS: {fps}")

        except Empty:
            pass

        self.root.after(33, self.update_camera_display)  # ~30 FPS update

    def update_log_display(self):
        """Update log display from queue."""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.add_log(message)
        except Empty:
            pass

        self.root.after(100, self.update_log_display)

    def add_log(self, message):
        """Add a message to the log display."""
        self.log_text.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def generate_gif(self):
        """Generate GIF by calling src/generate_gif.py script."""
        if not self.last_keypoint_path or not os.path.exists(self.last_keypoint_path):
            # Try to find the most recent keypoint file
            keypoint_dir = os.path.join(self.save_dir_var.get(), "keypoint")
            if os.path.exists(keypoint_dir):
                files = sorted(
                    [f for f in os.listdir(keypoint_dir) if f.endswith(".json")],
                    key=lambda x: os.path.getmtime(os.path.join(keypoint_dir, x)),
                    reverse=True,
                )
                if files:
                    self.last_keypoint_path = os.path.join(keypoint_dir, files[0])

        if not self.last_keypoint_path or not os.path.exists(self.last_keypoint_path):
            messagebox.showinfo(
                "Info", "No keypoint file found. Record a session first."
            )
            return

        self.status_var.set("Generating GIF...")
        self.add_log(
            f"Generating GIF from: {os.path.basename(self.last_keypoint_path)}"
        )

        # Run generate_gif.py script in a thread to avoid blocking UI
        def generate_thread():
            try:
                # Path to generate_gif.py script
                script_path = os.path.join(os.path.dirname(__file__), "generate_gif.py")

                # Calculate expected GIF output path
                base_dir = os.path.dirname(os.path.dirname(self.last_keypoint_path))
                gif_dir = os.path.join(base_dir, "gif")
                filename = os.path.splitext(os.path.basename(self.last_keypoint_path))[
                    0
                ]
                expected_gif_path = os.path.join(gif_dir, f"{filename}.gif")

                # Run the script with the keypoint path as argument
                result = subprocess.run(
                    [sys.executable, script_path, self.last_keypoint_path],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                )

                if result.returncode == 0:
                    self.log_queue.put(f"GIF saved: {expected_gif_path}")

                    # Load and display GIF if it exists
                    if os.path.exists(expected_gif_path):
                        self.load_gif(expected_gif_path)
                else:
                    error_msg = (
                        result.stderr.strip() if result.stderr else "Unknown error"
                    )
                    self.log_queue.put(f"GIF generation failed: {error_msg}")

            except Exception as e:
                self.log_queue.put(f"GIF generation error: {str(e)}")
            finally:
                self.status_var.set("Ready")

        threading.Thread(target=generate_thread, daemon=True).start()

    def run_gif_cli(self):
        """Open the generate_gif.py CLI in a new terminal window."""
        script_path = os.path.join(os.path.dirname(__file__), "generate_gif.py")

        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Script not found: {script_path}")
            return

        try:
            # On Windows, open a new cmd window and run the script
            subprocess.Popen(
                f'start cmd /k "{sys.executable}" "{script_path}"',
                shell=True,
                cwd=project_root,
            )
            self.add_log("Opened GIF CLI in new terminal window")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open CLI: {str(e)}")

    def load_gif(self, gif_path):
        """Load and animate a GIF in the preview area."""
        try:
            gif = Image.open(gif_path)
            self.gif_frames = []

            try:
                while True:
                    # Resize frame
                    frame = gif.copy()
                    frame.thumbnail((320, 150), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(frame)
                    self.gif_frames.append(photo)
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass

            if self.gif_frames:
                self.gif_index = 0
                self.animate_gif()

        except Exception as e:
            self.add_log(f"Error loading GIF: {str(e)}")

    def animate_gif(self):
        """Animate the GIF display."""
        if self.gif_frames:
            # Ensure index is within bounds (handles race condition if gif_frames is reloaded)
            if self.gif_index >= len(self.gif_frames):
                self.gif_index = 0
            self.gif_label.configure(image=self.gif_frames[self.gif_index], text="")
            self.gif_label.image = self.gif_frames[self.gif_index]
            self.gif_index = (self.gif_index + 1) % len(self.gif_frames)
            self.root.after(100, self.animate_gif)

    def on_close(self):
        """Handle window close event."""
        if self.camera_thread:
            self.camera_thread.stop()
            if self.camera_thread.is_alive():
                self.camera_thread.join(timeout=2)

        self.root.destroy()

    def run(self):
        """Start the application."""
        # Start camera on launch
        self.root.after(100, self.start_camera)
        self.root.mainloop()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    app = MoveUpApp()
    app.run()
