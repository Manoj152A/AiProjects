const examVideo = document.getElementById('exam-video');
const warningMessage = document.getElementById('warning-message');
const canvasOverlay = document.createElement('canvas');
const ctxOverlay = canvasOverlay.getContext('2d');

// Set canvas overlay size to match the video size
canvasOverlay.width = examVideo.width;
canvasOverlay.height = examVideo.height;
canvasOverlay.style.position = 'absolute';
canvasOverlay.style.top = examVideo.offsetTop + 'px';
canvasOverlay.style.left = examVideo.offsetLeft + 'px';
canvasOverlay.style.zIndex = '10';  // Ensure the canvas is on top of the video
canvasOverlay.style.pointerEvents = 'none';  // Disable interactions with the canvas

// Append the canvas to the same parent container as the video
examVideo.parentElement.appendChild(canvasOverlay);

// Access the webcam and stream it to the video element
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        examVideo.srcObject = stream;
    })
    .catch((error) => {
        console.log('Error accessing webcam:', error);
    });

// Function to capture image and check for person detection and recognition
function checkPerson() {
    const canvas = document.createElement('canvas');
    canvas.width = examVideo.width;
    canvas.height = examVideo.height;
    const context = canvas.getContext('2d');
    context.drawImage(examVideo, 0, 0, canvas.width, canvas.height);
    
    // Convert the image to a blob and send it to the server for face recognition
    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append('webcam', blob, 'webcam.jpg');

        fetch('/check_person', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            // Display warning message if the face is not recognized
            if (data.recognized) {
                warningMessage.innerText = "";  // Clear warning
            } else {
                warningMessage.innerText = data.message;
                warningMessage.style.color = "red";  // Set warning color
            }

            // Draw the face-tracking box if a face is detected
            if (data.face_box) {
                const { x1, y1, x2, y2 } = data.face_box;
                ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);  // Clear previous box
                ctxOverlay.strokeStyle = "green";  // Set box color
                ctxOverlay.lineWidth = 2;  // Set box thickness
                ctxOverlay.strokeRect(x1, y1, x2 - x1, y2 - y1);  // Draw the box
            } else {
                ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);  // Clear box if no face
            }
        })
        .catch((error) => {
            console.log('Error in verifying the person:', error);
        });
    });
}

// Run the face check every 1 second
setInterval(checkPerson, 1000);
