# Emo3D - Text to Facial Expression

This application converts text descriptions into facial expressions using a 3D face model. It uses CLIP for text understanding and a custom model to generate facial expressions.

## Prerequisites

- Python 3.11 or later
- pip (Python package installer)
- A modern web browser

## Installation

1. Clone or download this repository
2. Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server (for text processing):

```bash
python server.py
```

You should see output like:

```
Server starting in directory: /path/to/your/directory
 * Running on http://0.0.0.0:5003
 * Debug mode: on
```

2. In a new terminal window, start the HTTP server (for serving the web interface):

```bash
python -m http.server 8000
```

3. Open your web browser and go to:

```
http://localhost:8000/webgl_morphtargets_face.html
```

## Using the Application

1. You'll see a 3D face model in the center of the screen
2. Above the face, there's a text input box and an "Apply Expression" button
3. Enter a text description of the expression you want (e.g., "slightly surprise", "happy", "sad")
4. Click the "Apply Expression" button
5. The face will change to match the expression you described
6. Try these example expressions:

- "slightly surprise"
- "happy"
- "sad"
- "angry"
- "smiling"
- "laughing"
- "worried"

## Files

- `webgl_morphtargets_face.html`: The main web interface
- `server.py`: The Flask server for processing text
- `clip_embedding.py`: Handles CLIP text embedding generation
- `generate_blendshapes.py`: Generates facial expressions from embeddings
- `requirements.txt`: List of required Python packages

## Credits & Acknowledgments

- Built with [Three.js](https://github.com/mrdoob/three.js/blob/master/examples/webgl_morphtargets_face.html) - A powerful 3D graphics library for the web
- Face model provided by [Face Cap](https://www.bannaflak.com/face-cap)
- Text understanding powered by [OpenAI CLIP](https://github.com/openai/CLIP)
