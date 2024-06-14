import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import threading
from facenet_pytorch import MTCNN

app = FastAPI()

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Create a lock for thread-safe operations
lock = threading.Lock()

def detect_faces(frame):
    boxes, probs = mtcnn.detect(frame)
    return boxes, probs

def put_centered_text(frame, text, y, font_scale=1, color=(0, 255, 0), thickness=1):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width = text_size[0]
    frame_width = frame.shape[1]
    x = (frame_width - text_width) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    face_boxes, probs = detect_faces(img)
    num_faces = len(face_boxes) if face_boxes is not None else 0

    if face_boxes is not None:
        for box, prob in zip(face_boxes, probs):
            if prob is not None:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if num_faces == 1:
        put_centered_text(img, 'One face detected, real-time liveness prediction in process.', 60, font_scale=1, color=(0, 255, 0), thickness=2)
    elif num_faces > 1:
        put_centered_text(img, 'Multiple faces in the view. Real-time liveness works on one face only.', 60, font_scale=1, color=(0, 0, 255), thickness=2)
    else:
        put_centered_text(img, 'No face detected in the view. One face is required for real-time liveness.', 60, font_scale=1, color=(0, 0, 255), thickness=2)

    # Display the number of faces on the top right corner
    cv2.putText(img, f'Total faces: {num_faces}', (img.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
