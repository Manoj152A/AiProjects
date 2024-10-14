from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import wave
import pyaudio
import time
import numpy as np
from threading import Thread
from face_utils.face_recognition import FaceRecognition
import psycopg2

app = Flask(__name__)

# Database connection parameters
DB_HOST = "localhost"
DB_NAME = "exam_proctoring"
DB_USER = "postgres"
DB_PASS = "Password"  # Replace with your actual database password

# Global objects for face recognition and audio capture
face_recognition = None
audio_capture = None
audio_stream = None  # Stream object for audio
video_capture = None
audio_file = "captured/exam_audio.wav"
video_file_path = "captured/exam_video.mp4"  # Path to save the video
frames = []
flagged_events = []
record_audio_thread = None  # Thread for audio recording
recording = False  # Flag to control audio recording

# Video buffer to store frames around flagged events
recording_duration = 20  # Duration to record additional frames for flagged events (20 seconds)
fps = 20  # Frames per second for video recording

# Store video during the exam
video_writer = None

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
    face_recognition = FaceRecognition(reference_image_paths=[img_path])
    return redirect(url_for('exam_page'))

# Start video capture
def start_video_capture():
    global video_capture, video_writer
    video_capture = cv2.VideoCapture(0)  # Open the webcam
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return False
    
    # Setup video writer for continuous recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(video_file_path, fourcc, fps, (640, 480))
    
    print("Video capture started.")
    return True  # Indicate video capture started

# Stop video capture
def stop_video_capture():
    global video_capture, video_writer
    if video_writer:
        video_writer.release()
        print("Video writer released.")
        video_writer = None  # Clear the writer to prevent reuse
    if video_capture:
        video_capture.release()
        print("Video capture stopped.")
        video_capture = None  # Clear the capture object

# Dummy exam page
@app.route('/exam')
def exam_page():
    if not start_video_capture():  # Start video capture
        return "Failed to open camera.", 500  # Return error if camera fails
    start_audio_capture()  # Start capturing audio when the exam page is accessed
    return render_template('exam.html')

# Route to check the person during the exam
@app.route('/check_person', methods=['POST'])
def check_person():
    global flagged_events, video_writer
    try:
        img_data = request.files['webcam']
        img_path = os.path.join('captured', 'exam_image.jpg')
        img_data.save(img_path)
        captured_image = cv2.imread(img_path)

        # Write the current frame to the video
        if video_writer:
            video_writer.write(captured_image)

        # Check face tracking and recognition
        if face_recognition is None:
            print("Face recognition object not initialized.")
            return jsonify({"recognized": False, "message": "Face recognition not initialized."})

        results = face_recognition.recognize_faces(captured_image)

        # Check for flagged events based on recognition results
        for result in results:
            if result['flagged']:
                flagged_events.append({"event": result['event'], "timestamp": time.time()})
                print("Flagged: " + result['event'])

        if not results:
            flagged_events.append({"event": "No face detected", "timestamp": time.time()})
            print("Flagged: No face detected")
            return jsonify({"redirect": url_for('report_page')})

        print("Faces detected and processed.")
        return jsonify({"recognized": True, "message": "Faces processed successfully", "status": "good"})

    except Exception as e:
        print(f"Error in capturing or recognizing image: {e}")
        return jsonify({"recognized": False, "message": "Error in recognition process"})

# Route to submit the exam
@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    print("Submitting exam...")  # Debugging output
    # Stop capturing audio and video regardless of flagged activities
    stop_audio_capture()
    stop_video_capture()

    # Save exam session data in the database
    if save_exam_session():  # Ensure session is saved successfully
        print("Exam session saved successfully.")
    else:
        print("Failed to save exam session.")

    # Check for flagged activities
    if flagged_events:
        # Redirect to report page if there are flagged activities
        return redirect(url_for('report_page'))
    else:
        # If no flagged activities, redirect to thank you page
        return redirect(url_for('thank_you_page'))

# Save exam session data to the database
def save_exam_session():
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        cur = conn.cursor()
        print("Database connection established.")

        # Insert a new exam session
        cur.execute("INSERT INTO exam_sessions (user_id, video_path, audio_path) VALUES (%s, %s, %s) RETURNING id;",
                    (1, video_file_path, audio_file))  # Replace user_id with actual user_id
        session_id = cur.fetchone()[0]
        
        # Insert flagged events
        for event in flagged_events:
            cur.execute("INSERT INTO flagged_events (session_id, event, timestamp) VALUES (%s, %s, %s);",
                        (session_id, event['event'], event['timestamp']))
        
        conn.commit()
        print("Exam session saved to the database.")
        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"Error saving exam session to database: {e}")
        return False

# Route to display the report page
@app.route('/report')
def report_page():
    global flagged_events, video_file_path
    no_flagged_events = len(flagged_events) == 0  # Check if there are no flagged events
    return render_template('report.html', flagged_events=flagged_events, no_flagged_events=no_flagged_events)

# Start audio capture
def start_audio_capture():
    global audio_capture, audio_stream, frames, recording, record_audio_thread
    recording = True  # Set recording flag
    audio_capture = pyaudio.PyAudio()
    audio_stream = audio_capture.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)  # Store stream object
    frames = []

    def record_audio():
        while recording:  # Use recording flag
            data = audio_stream.read(1024)  # Use audio_stream
            frames.append(data)

    record_audio_thread = Thread(target=record_audio)
    record_audio_thread.start()

# Stop audio capture
def stop_audio_capture():
    global audio_capture, audio_stream, recording
    recording = False  # Set recording flag to False
    if audio_stream:  # Check if the stream exists
        audio_stream.stop_stream()  # Stop the stream
        audio_stream.close()  # Close the stream
        audio_stream = None  # Clear the stream object
    if audio_capture:  # Check if audio_capture exists
        audio_capture.terminate()  # Terminate PyAudio instance
        audio_capture = None  # Clear the PyAudio instance

# Route for thank you page
@app.route('/thank_you')
def thank_you_page():
    return render_template('thank_you.html')

if __name__ == '__main__':
    if not os.path.exists('captured'):
        os.makedirs('captured')
    app.run(debug=True)
