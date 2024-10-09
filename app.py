from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
from utils.face_recognition import FaceRecognition

app = Flask(__name__)

# Global object for face recognition
face_recognition = None

@app.route('/')
def index():
    return redirect(url_for('capture_page'))

# First page to capture the reference image
@app.route('/capture')
def capture_page():
    return render_template('capture.html')

# Route to capture and save an image from the webcam
@app.route('/save_capture', methods=['POST'])
def save_capture():
    global face_recognition
    img_data = request.files['webcam']
    img_path = os.path.join('captured', 'reference_image.jpg')
    img_data.save(img_path)
    # Initialize face recognition object with reference image
    face_recognition = FaceRecognition(reference_image_path=img_path)
    return redirect(url_for('exam_page'))

# Dummy exam page
@app.route('/exam')
def exam_page():
    return render_template('exam.html')

# Route to check the person during the exam
@app.route('/check_person', methods=['POST'])
def check_person():
    try:
        img_data = request.files['webcam']
        img_path = os.path.join('captured', 'exam_image.jpg')
        img_data.save(img_path)
        captured_image = cv2.imread(img_path)
        
        # Check face tracking and recognition
        result, face_detected, face_box = face_recognition.recognize_and_track_face(captured_image)
        
        if not face_detected:
            return jsonify({"recognized": False, "message": "No face detected"})
        
        # Return face bounding box coordinates
        face_box_data = {
            'x1': int(face_box[0]),
            'y1': int(face_box[1]),
            'x2': int(face_box[2]),
            'y2': int(face_box[3])
        }
        return jsonify({"recognized": result, "message": "Face recognized" if result else "Face not recognized", "face_box": face_box_data})
    except Exception as e:
        print(f"Error in capturing or recognizing image: {e}")
        return jsonify({"recognized": False, "message": "Error in recognition process"})

if __name__ == '__main__':
    if not os.path.exists('captured'):
        os.makedirs('captured')
    app.run(debug=True)
