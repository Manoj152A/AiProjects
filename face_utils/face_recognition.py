import cv2
import numpy as np
import mediapipe as mp
import face_recognition
import os

class FaceRecognition:
    def __init__(self, reference_image_paths):
        self.reference_image_paths = reference_image_paths
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
        self.reference_embeddings = self.load_reference_embeddings()

    def load_reference_embeddings(self):
        embeddings = []
        for image_path in self.reference_image_paths:
            img = face_recognition.load_image_file(image_path)
            img_encoding = face_recognition.face_encodings(img)

            if img_encoding:
                embeddings.append(img_encoding[0])
            else:
                raise ValueError(f"No face detected in the reference image: {image_path}")
        return embeddings

    def recognize_faces(self, captured_image):
        img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.reference_embeddings, face_encoding)

            if True in matches:
                label = "Recognized"
                results.append({"face_box": (left, top, right, bottom), "flagged": False})
            else:
                label = "Unknown"
                results.append({"face_box": (left, top, right, bottom), "flagged": True, "event": "Unknown face detected"})

        return results

    def is_face_out_of_focus(self, image, face_box):
        (x1, y1, x2, y2) = face_box.astype(int)
        face_roi = image[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return True  # No face region

        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        focus_measure = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        focus_threshold = 100  # Adjust this value for your needs
        min_face_size = 30  # Minimum size for the face to be considered

        # Check face size
        face_width = x2 - x1
        face_height = y2 - y1
        if face_width < min_face_size or face_height < min_face_size:
            return True  # Face too small

        return focus_measure < focus_threshold

    def recognize_and_track_face(self, captured_image):
        img_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)

        if not results.detections:
            return False, False, None  # No face detected

        # Assume first detected face is the person to track
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = captured_image.shape
        x1, y1, x2, y2 = int(bboxC.xmin * w), int(bboxC.ymin * h), int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)
        face_box = np.array([x1, y1, x2, y2])

        # Check if the face is out of focus
        if self.is_face_out_of_focus(captured_image, face_box):
            return False, True, "out_of_focus"  # Flag as out of focus

        # Perform face recognition
        recognition_results = self.recognize_faces(captured_image)
        return True, True, face_box, recognition_results  # Return recognition results
