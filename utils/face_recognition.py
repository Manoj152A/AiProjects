import cv2
from insightface.app import FaceAnalysis
import numpy as np

class FaceRecognition:
    def __init__(self, reference_image_path):
        self.reference_image_path = reference_image_path

        # Initialize RetinaFace from InsightFace
        self.face_detector = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.face_detector.prepare(ctx_id=-1)  # Use CPU

        # Get face embedding of the reference image using ArcFace
        self.reference_embedding = self.get_face_embedding(self.reference_image_path)

    def get_face_embedding(self, image_path):
        # Use ArcFace to get the face embedding of an image
        img = cv2.imread(image_path)
        faces = self.face_detector.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the reference image.")
        
        # Extract the first detected face and get its embedding
        face = faces[0]
        return face.normed_embedding

    def recognize_and_track_face(self, captured_image):
        # Detect faces using RetinaFace
        faces = self.face_detector.get(captured_image)

        if len(faces) == 0:
            print("No face detected.")
            return False, False, None  # No face detected

        # Assume first detected face is the person to track
        face = faces[0]
        current_embedding = face.normed_embedding

        # Get face bounding box
        face_box = face.bbox

        # Compare the current embedding with the reference embedding
        similarity = np.linalg.norm(self.reference_embedding - current_embedding)
        
        print(f"Similarity score: {similarity}")

        # Set a threshold for similarity; lower is more similar
        if similarity < 0.6:
            return True, True, face_box  # Recognized and face detected
        else:
            return False, True, face_box  # Not recognized but face detected
