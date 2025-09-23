import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import os

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Model and Video Setup ---
# The trained model file from your training run.
model_path = "C:/Users/arano/Desktop/YOLO_parking project/best.pt"
model = YOLO(model_path)

# Your pre-recorded video file.
video_file = 'carPark.mp4'

def generate_frames():
    """
    Generator function to stream video frames to the web page.
    This creates a live video feed with the YOLO overlay.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_file}'.")
        return

    while True:
        # Loop the video if it ends.
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the frame to get detection results.
        results = model(frame, conf=0.5)

        # The 'plot()' method automatically draws bounding boxes, labels, and confidence scores.
        processed_frame = results[0].plot()

        # Encode the frame to JPEG for streaming.
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """
    API endpoint to stream the live video with real-time analysis.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    """
    API endpoint to return the current parking status as a JSON object.
    """
    # Create a temporary video capture object to read a single frame.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return jsonify({"error": "Video file not found"}), 500
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Could not read frame"}), 500
    
    # Perform inference on the single frame to get a snapshot of the parking lot.
    results = model(frame, conf=0.5)
    
    # Count the number of 'car' and 'free' detections.
    car_count = 0
    free_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == 'car':
                car_count += 1
            elif model.names[class_id] == 'free':
                free_count += 1
    
    # Return the data as JSON.
    return jsonify({
        "free_spots": free_count,
        "occupied_spots": car_count,
        "total_spots": car_count + free_count
    })

if __name__ == '__main__':
    # You must have Flask and ultralytics installed.
    # Make sure 'best.pt' and 'carPark.mp4' are in the same directory.
    print("API server is running. Go to http://127.0.0.1:5000/video_feed for the video stream.")
    print("Go to http://127.0.0.1:5000/api/status for the JSON output.")
    app.run(host='0.0.0.0', port=5000, debug=True)
