AI Parking Management System
This repository contains a full-stack, AI-powered parking management system designed as a hackathon project. The system uses a custom-trained YOLOv8n model to accurately detect and count available and occupied parking spaces in real-time from a video feed.

The project is built to be modular and scalable, with a clear separation between the backend (computer vision and API) and the frontend.

Features
Custom-Trained YOLOv8n Model: A lightweight but highly accurate model trained on a custom dataset to identify 'cars' and 'free' parking spaces.

Real-Time Video Analysis: Processes a pre-recorded video or a live camera feed to provide instant updates on parking availability.

Backend API: A simple Flask server provides two endpoints for easy integration with a web or mobile app:

/video_feed: Streams the live video with real-time bounding boxes and status overlays.

/api/status: Provides a JSON response with the total count of free and occupied spots.

Scalable Architecture: The use of a trained model eliminates the need for manual spot configuration, making the system highly scalable for different parking lots.

Getting Started
Prerequisites
Python 3.8 or higher

pip package manager

Installation
Clone this repository.

Navigate to the project directory.

Install the required Python packages:

pip install -r requirements.txt


Running the Application
Train the Model: The training script will use your custom dataset to generate the necessary model weights (best.pt).

Run the Backend Server: Start the Flask backend server, which handles all the computer vision and API logic.

View the App: Open a web browser and navigate to the provided URL to see the live parking management system in action.

Project Structure
train_model.py: The script used to train the YOLOv8 model.

inference_script.py: A script for running and testing the trained model on a video.

api_server.py: The backend server for the web application.

my_dataset/: The folder containing all the images and labels for model training.

runs/: The directory where all training results, including the final model weights, are saved.
