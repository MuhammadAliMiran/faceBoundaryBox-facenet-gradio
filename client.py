import gradio as gr
import cv2
import time
import requests
import numpy as np

API_URL = "http://localhost:8000/detect/"

def call_detection_api(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
    if response.status_code == 200:
        nparr = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return frame

def capture_image():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the camera index if you have multiple cameras
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set the buffer size to 1 for lower latency

    # Initialize variables for FPS calculation
    prev_time = 0
    curr_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        frame = call_detection_api(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        yield frame

    cap.release()

iface = gr.Interface(
    fn=capture_image,
    inputs=None,
    outputs=gr.Image(label="Output", streaming=True),
    title="Camera Capture with Face Detection",
    description="Live video feed from the webcam with face detection."
)

# Launch the Gradio interface
iface.launch(debug=True)
