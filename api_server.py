import cv2
from flask import Flask, Response, jsonify
# Import CORS to allow the web app (on a different origin) to connect
from flask_cors import CORS 
from ultralytics import YOLO
import os

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all routes (*) to ensure the web app can communicate
CORS(app) 

# --- Model and Video Setup ---
# NOTE: Ensure 'best.pt' and 'carPark.mp4' are in the same directory as this script.
model_path = "best.pt" 
video_file = 'carPark.mp4' 

model = None
try:
    # Attempt to load the YOLO model
    model = YOLO(model_path)
    print(f"YOLO model loaded successfully from: {model_path}")
except Exception as e:
    print(f"ERROR: Could not load YOLO model from {model_path}. Ensure the path is correct and dependencies are installed.")
    print(f"Details: {e}")

def generate_frames():
    """
    Generator function to stream video frames to the web page.
    This creates a live video feed with the YOLO overlay (MJPEG stream).
    """
    # Use a local path for the video file
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
            print("Warning: Failed to read frame or end of video.")
            break
        
        processed_frame = frame
        if model:
            # Perform inference on the frame to get detection results.
            # verbose=False reduces console spam during inference
            results = model(frame, conf=0.5, verbose=False) 
            # The 'plot()' method automatically draws bounding boxes, labels, and confidence scores.
            processed_frame = results[0].plot()
        
        # Encode the frame to JPEG for streaming.
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    # Release the capture object when done
    cap.release()

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
    This is polled by the web application for real-time counts.
    """
    if not model:
        return jsonify({"error": "YOLO model failed to load. Check console for details."}), 500

    # Create a temporary video capture object to read a single frame.
    # NOTE: Re-opening the video file for every status request is inefficient. 
    # For a production app, you should maintain a global, continuously processing video thread/object.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return jsonify({"error": f"Video file '{video_file}' not found"}), 500
    
    # Read a single frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Could not read frame from video file"}), 500
    
    # Perform inference on the single frame to get a snapshot of the parking lot.
    results = model(frame, conf=0.5, verbose=False)
    
    # Count the number of 'car' and 'free' detections.
    car_count = 0
    free_count = 0
    
    # NOTE: Assuming your model's class names are 'car' and 'free' based on common YOLO parking lot usage.
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id)
            
            if class_name == 'car':
                car_count += 1
            elif class_name == 'free':
                free_count += 1
    
    # Return the data as JSON.
    return jsonify({
        "free_spots": free_count,
        "occupied_spots": car_count,
        "total_spots": car_count + free_count
    })

if __name__ == '__main__':
    print("\n--- Starting SwiftSlot API Server ---")
    print(f"Video file being analyzed: {video_file}")
    print("API Status Endpoint: http://127.0.0.1:5000/api/status")
    print("Video Stream Endpoint: http://127.0.0.1:5000/video_feed")
    print("!!! Ensure 'best.pt' and 'carPark.mp4' are in the same folder. !!!\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
