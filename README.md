# Real-Time Hand Gesture Recognition with MediaPipe and OpenCV

## Overview
This project implements a real-time hand gesture recognition system using MediaPipe's Hand Tracking and OpenCV. It detects hand gestures such as "Thumbs Up," "Cross," "Palm," and "Wave" from a live video feed by analyzing the position of hand landmarks. The program provides feedback by displaying the recognized gesture name on the video feed.

## Key Features
- **Real-Time Hand Tracking**: Uses MediaPipe to detect and track hand landmarks.
- **Gesture Recognition**: Detects gestures based on finger positions:
  - **Thumbs Up (All OK)**
  - **Cross (Not OK)**
  - **Palm (Stop)**
  - **Wave (Help)**
- **Live Video Feed**: Displays recognized gestures in real-time using OpenCV.

## Technical Stack
- **Languages**: Python
- **Libraries**: OpenCV, MediaPipe, NumPy
- **Gesture Recognition**: Based on hand landmark positions

## Getting Started

### Prerequisites
- Python 3.x
- MediaPipe
- OpenCV
- NumPy

Example Usage
The project runs a live video feed to detect and display the recognized hand gesture on the screen. Press ESC to exit the video feed.
