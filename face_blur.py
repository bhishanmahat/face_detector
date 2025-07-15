"""
Real-time Face Detection with Face Blurring using OpenCV
An advanced implementation for detecting faces from webcam feed
with interactive face blurring capabilities and optimized performance.
"""

import cv2
import sys
import logging
import os
from datetime import datetime
from typing import Tuple, List, Optional
import numpy as np

# Configure file logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"face_blur_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class FaceBlurDetector:
    """
    A face detection class with blurring capabilities using OpenCV's DNN face detector.
    Provides real-time face detection with bounding box visualization and face blurring.
    """

    def __init__(self, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize the face detector with DNN model.

        Args:
            confidence_threshold: Minimum confidence for face detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = self._load_dnn_model()
        self.blur_enabled = False
        self.blur_intensity = 15  # Blur kernel size (must be odd)
        self.blur_type = "gaussian"  # 'gaussian', 'motion', 'pixelate'

    def _load_dnn_model(self) -> cv2.dnn.Net:
        """Load the DNN face detection model."""
        try:
            # Download DNN model files if they don't exist
            self._download_dnn_model()

            prototxt_path = "opencv_face_detector.pbtxt"
            model_path = "opencv_face_detector_uint8.pb"

            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                logger.info("DNN face detector loaded successfully")
                return net
            else:
                logger.warning(
                    "DNN model files not found, falling back to Haar cascade"
                )
                return None

        except Exception as e:
            logger.error(f"Error loading face detection model: {e}")
            return None

    def _download_dnn_model(self):
        """Download DNN model files from reliable source."""
        import urllib.request

        # File paths
        prototxt_path = "opencv_face_detector.pbtxt"
        model_path = "opencv_face_detector_uint8.pb"

        # LearnOpenCV repository URLs (reliable source)
        base_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models"
        prototxt_url = f"{base_url}/opencv_face_detector.pbtxt"
        model_url = f"{base_url}/opencv_face_detector_uint8.pb"

        try:
            # Download prototxt configuration file
            if not os.path.exists(prototxt_path):
                logger.info("Downloading DNN prototxt file...")
                urllib.request.urlretrieve(prototxt_url, prototxt_path)
                logger.info("Prototxt file downloaded successfully")

            # Download model weights file
            if not os.path.exists(model_path):
                logger.info("Downloading DNN model file (this may take a moment)...")
                urllib.request.urlretrieve(model_url, model_path)
                logger.info("Model file downloaded successfully")

        except Exception as e:
            logger.error(f"Error downloading DNN model files: {e}")
            logger.info("Will use Haar cascade instead")

    def detect_faces_dnn(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN model.

        Args:
            frame: Input frame from webcam

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.net is None:
            return []

        h, w = frame.shape[:2]

        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def detect_faces_haar(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar cascade classifier.

        Args:
            frame: Input frame from webcam

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return faces

    def detect_faces(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using the best available method.

        Args:
            frame: Input frame from webcam

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if self.net is not None:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)

    def apply_gaussian_blur(
        self, face_region: np.ndarray, intensity: int
    ) -> np.ndarray:
        """Apply Gaussian blur to face region."""
        # Ensure intensity is odd for Gaussian blur
        if intensity % 2 == 0:
            intensity += 1
        return cv2.GaussianBlur(face_region, (intensity, intensity), 0)

    def apply_motion_blur(self, face_region: np.ndarray, intensity: int) -> np.ndarray:
        """Apply motion blur to face region."""
        # Create motion blur kernel
        kernel = np.zeros((intensity, intensity))
        kernel[int((intensity - 1) / 2), :] = np.ones(intensity)
        kernel = kernel / intensity
        return cv2.filter2D(face_region, -1, kernel)

    def apply_pixelate_blur(
        self, face_region: np.ndarray, intensity: int
    ) -> np.ndarray:
        """Apply pixelation effect to face region."""
        h, w = face_region.shape[:2]

        # Resize down and then up to create pixelation effect
        pixel_size = max(1, intensity // 3)
        temp = cv2.resize(
            face_region,
            (w // pixel_size, h // pixel_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    def blur_faces(
        self, frame: cv2.Mat, faces: List[Tuple[int, int, int, int]]
    ) -> cv2.Mat:
        """
        Apply blur effect to detected faces.

        Args:
            frame: Input frame
            faces: List of face bounding boxes

        Returns:
            Frame with blurred faces
        """
        if not self.blur_enabled:
            return frame

        # Create a copy of the frame to work with
        blurred_frame = frame.copy()

        for x, y, w, h in faces:
            # Ensure coordinates are within frame bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            # Extract face region
            face_region = frame[y : y + h, x : x + w]

            if face_region.size > 0:
                # Apply selected blur type
                if self.blur_type == "gaussian":
                    blurred_face = self.apply_gaussian_blur(
                        face_region, self.blur_intensity
                    )
                elif self.blur_type == "motion":
                    blurred_face = self.apply_motion_blur(
                        face_region, self.blur_intensity
                    )
                elif self.blur_type == "pixelate":
                    blurred_face = self.apply_pixelate_blur(
                        face_region, self.blur_intensity
                    )
                else:
                    blurred_face = self.apply_gaussian_blur(
                        face_region, self.blur_intensity
                    )

                # Replace the face region with blurred version
                blurred_frame[y : y + h, x : x + w] = blurred_face

        return blurred_frame

    def draw_faces(
        self, frame: cv2.Mat, faces: List[Tuple[int, int, int, int]]
    ) -> cv2.Mat:
        """
        Draw bounding boxes around detected faces.

        Args:
            frame: Input frame
            faces: List of face bounding boxes

        Returns:
            Frame with drawn bounding boxes
        """
        for x, y, w, h in faces:
            # Choose color based on blur status
            color = (
                (0, 0, 255) if self.blur_enabled else (0, 255, 0)
            )  # Red if blurred, green if not

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Add status text
            status_text = "Blurred" if self.blur_enabled else "Face"
            cv2.putText(
                frame,
                status_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        return frame

    def toggle_blur(self):
        """Toggle face blurring on/off."""
        self.blur_enabled = not self.blur_enabled
        status = "ON" if self.blur_enabled else "OFF"
        logger.info(f"Face blurring: {status}")

    def change_blur_intensity(self, delta: int):
        """Change blur intensity."""
        self.blur_intensity = max(5, min(51, self.blur_intensity + delta))
        # Ensure intensity is odd for Gaussian blur
        if self.blur_intensity % 2 == 0:
            self.blur_intensity += 1
        logger.info(f"Blur intensity: {self.blur_intensity}")

    def cycle_blur_type(self):
        """Cycle through different blur types."""
        blur_types = ["gaussian", "motion", "pixelate"]
        current_index = blur_types.index(self.blur_type)
        self.blur_type = blur_types[(current_index + 1) % len(blur_types)]
        logger.info(f"Blur type: {self.blur_type}")

    def draw_ui_info(
        self, frame: cv2.Mat, faces: List[Tuple[int, int, int, int]]
    ) -> cv2.Mat:
        """Draw UI information on frame."""
        # Face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Blur status
        blur_status = "ON" if self.blur_enabled else "OFF"
        blur_color = (0, 0, 255) if self.blur_enabled else (0, 255, 0)
        cv2.putText(
            frame,
            f"Blur: {blur_status}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            blur_color,
            2,
        )

        # Blur type and intensity (only if blur is enabled)
        if self.blur_enabled:
            cv2.putText(
                frame,
                f"Type: {self.blur_type.title()}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                f"Intensity: {self.blur_intensity}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Controls info
        controls_text = [
            "Controls:",
            "(B) - Toggle blur",
            "(T) - Change blur type",
            "(+/-) - Adjust intensity",
            "(Q) - Quit",
        ]

        for i, text in enumerate(controls_text):
            cv2.putText(
                frame,
                text,
                (10, frame.shape[0] - 120 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return frame


def main():
    """Main function to run real-time face detection with blurring."""

    # Initialize face blur detector
    detector = FaceBlurDetector(confidence_threshold=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Error: Could not open webcam")
        sys.exit(1)

    # Set webcam properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info(f"Starting face detection with blurring. Log file: {log_path}")
    logger.info("Interactive Controls:")
    logger.info("  B - Toggle face blurring")
    logger.info("  T - Change blur type (Gaussian/Motion/Pixelate)")
    logger.info("  + - Increase blur intensity")
    logger.info("  - - Decrease blur intensity")
    logger.info("  Q - Quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                logger.error("Error: Could not read frame from webcam")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect faces
            faces = detector.detect_faces(frame)

            # Apply face blurring if enabled
            frame = detector.blur_faces(frame, faces)

            # Draw bounding boxes (after blurring to show boxes on blurred faces)
            if (
                not detector.blur_enabled
            ):  # Only show boxes when not blurring for cleaner look
                frame = detector.draw_faces(frame, faces)

            # Draw UI information
            frame = detector.draw_ui_info(frame, faces)

            # Display frame
            cv2.imshow("Face Detection with Blurring", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("b"):
                detector.toggle_blur()
            elif key == ord("t"):
                detector.cycle_blur_type()
            elif key == ord("+") or key == ord("="):
                detector.change_blur_intensity(2)
            elif key == ord("-"):
                detector.change_blur_intensity(-2)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Face detection with blurring stopped")


if __name__ == "__main__":
    main()
