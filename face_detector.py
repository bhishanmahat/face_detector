"""
Real-time Face Detection using OpenCV
An advanced implementation for detecting faces from webcam feed
with optimized performance and robust error handling.
"""

import cv2
import sys
import logging
import os
from datetime import datetime
from typing import Tuple, List, Optional

# Configure file logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),  # Optional: keep console output
    ],
)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    A face detection class using OpenCV's DNN face detector.
    Provides real-time face detection with bounding box visualization.
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
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add confidence text (if available)
            cv2.putText(
                frame,
                "Face",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return frame

def main():
    """Main function to run real-time face detection."""

    # Initialize face detector
    detector = FaceDetector(confidence_threshold=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Error: Could not open webcam")
        sys.exit(1)

    # Set webcam properties for optimal performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger.info(f"Starting face detection. Log file: {log_path}")
    logger.info("Press 'q' to quit.")

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

            # Draw bounding boxes
            frame = detector.draw_faces(frame, faces)

            # Add frame info
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display frame
            cv2.imshow("Face Detection", frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Face detection stopped")

if __name__ == "__main__":
    main()
