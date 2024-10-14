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

# Global objects for face recognition and audio capture
face_recognition = None
audio_capture = None
audio_stream = None  # Stream object for audio
video_capture = None
audio_file = "captured/exam_audio.wav"
video_file_path = "captured/exam_video.mp4"  # Path to save the video
frames = []
flagged_events = []

# Video buffer to store frames around flagged events
recording_duration = 20  # Duration to record additional frames for flagged events (20 seconds)
fps = 20  # Frames per second for video recording

# Store video during the exam
video_writer = None

# PostgreSQL database connection parameters
db_params = {
    "dbname": "exam_proctoring",
    "user": "postgres",
    "password": "your_password",  # Replace with your password
    "host": "localhost",
    "port": "5432"
}

def db_connect():
    """Connect to the PostgreSQL database."""
    conn = psycopg2.connect(**db_params)
    return conn

@app.route('/')
def index():
    return redirect(url_for('capture_page'))

# First page to capture the reference images
@app.route('/capture')
def capture_page():
    return render_template('capture.html')

# Route to capture and save images from the webcam
@app.route('/save_capture', methods=['POST'])
def save_capture():
    global face_recognition
    img_files = request.files.getlist('webcam')
    img_paths = []

    for img_data in img_files:
        img_path = os.path.join('captured', f'reference_image_{len(img_paths)}.jpg')
        img_data.save(img_path)
        img_paths.append(img_path)

    # Initialize face recognition object with reference images
    face_recognition = FaceRecognition(reference_image_paths=img_paths)
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
    # Stop capturing audio and video regardless of flagged activities
    stop_audio_capture()
    stop_video_capture()
    
    # Check for flagged activities
    if flagged_events:
        # Redirect to report page if there are flagged activities
        return redirect(url_for('report_page'))
    else:
        # If no flagged activities, redirect to thank you page
        return redirect(url_for('thank_you_page'))

# Route to display the report page
@app.route('/report')
def report_page():
    global flagged_events, video_file_path
    cut_clips()  # Cut clips based on flagged events
    no_flagged_events = len(flagged_events) == 0  # Check if there are no flagged events
    return render_template('report.html', flagged_events=flagged_events, no_flagged_events=no_flagged_events)

def cut_clips():
    global video_file_path
    if not os.path.exists(video_file_path) or os.path.getsize(video_file_path) == 0:
        print(f"Video file not found or is empty: {video_file_path}")
        return  # Exit if the video file does not exist or is empty

    import moviepy.editor as mpy
    try:
        video = mpy.VideoFileClip(video_file_path)

        for event in flagged_events:
            start_time = event['timestamp'] - 10  # Start 10 seconds before the event (10 seconds into the clip)
            end_time = event['timestamp'] + recording_duration  # End after the recording duration (20 seconds)

            if start_time < 0:
                start_time = 0  # Ensure start time is not negative

            # Cut the clip and save
            clip = video.subclip(start_time, end_time)
            clip_file_path = f"captured/event_{event['event']}_{int(event['timestamp'])}.mp4"
            clip.write_videofile(clip_file_path, codec="libx264")

        print("Clips have been created for flagged events.")

    except Exception as e:
        print(f"Error while cutting clips: {e}")

# Start audio capture
def start_audio_capture():
    global audio_capture, audio_stream, frames
    audio_capture = pyaudio.PyAudio()
    audio_stream = audio_capture.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)  # Store stream object
    frames = []

    def record_audio():
        while True:
            data = audio_stream.read(1024)  # Use audio_stream
            frames.append(data)

    Thread(target=record_audio).start()

# Stop audio capture
def stop_audio_capture():
    global audio_capture, audio_stream
    if audio_stream:  # Check if the stream exists
        audio_stream.stop_stream()  # Stop the stream
        audio_stream.close()  # Close the stream
        audio_stream = None  # Clear the stream object
    if audio_capture:  # Check if audio_capture exists
        audio_capture.terminate()  # Terminate PyAudio instance
        audio_capture = None  # Clear the PyAudio instance

@app.route('/end_exam')
def end_exam():
    stop_audio_capture()  # Stop audio capture
    stop_video_capture()  # Stop video capture
    audio_analysis_result = analyze_audio()  # Analyze the audio

    # Prepare the report with flagged events
    report = {"audio_analysis": audio_analysis_result, "flagged_events": flagged_events}
    flagged_events.clear()  # Clear events for the next exam
    return render_template('report.html', report=report)

def analyze_audio():
    # Load the audio file
    wf = wave.open(audio_file, 'rb')
    frames = wf.readframes(-1)
    sound_info = np.frombuffer(frames, dtype=np.int16)

    # Analyze audio data (e.g., check for loud sounds)
    volume_threshold = 1000  # Set a volume threshold
    loud_sound_detected = np.any(np.abs(sound_info) > volume_threshold)

    if loud_sound_detected:
        return {"message": "Loud sound detected during the exam."}
    else:
        return {"message": "No loud sounds detected."}

# Route for thank you page
@app.route('/thank_you')
def thank_you_page():
    return render_template('thank_you.html')

if __name__ == '__main__':
    if not os.path.exists('captured'):
        os.makedirs('captured')
    app.run(debug=True)
