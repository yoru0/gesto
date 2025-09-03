# Gesture Media Control

A hand gesture recognition system that controls media playback using your webcam.

## Features

- Play/Pause: Make a fist then open your hand
- Next Track: Two-finger swipe left  
- Previous Track: Two-finger swipe right
- Real-time gesture detection with confidence filtering

## Installation

### Prerequisites

- Python 3.8+
- Go 1.19+
- Webcam
- Windows 10/11

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yoru0/gestures.git
   cd gestures
   ```

2. Install Python dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   cd go
   go run .
   ```

## Usage

The application will open your camera and start detecting gestures:

- **Play/Pause**: Make a fist, then open your hand within 1 second
- **Next**: Show two fingers (peace sign) and swipe left
- **Previous**: Show two fingers (peace sign) and swipe right

Press 'q' in the camera window to quit.