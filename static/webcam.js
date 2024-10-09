const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureButton = document.getElementById('capture');
const form = document.getElementById('image-form');
const webcamInput = document.getElementById('webcam-input');

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.log('Error accessing webcam:', error);
    });

// Capture the image when the button is clicked
captureButton.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert the image to a blob and submit the form
    canvas.toBlob((blob) => {
        const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        webcamInput.files = dataTransfer.files;
        form.submit();  // Submit the form with the image file
    });
});
