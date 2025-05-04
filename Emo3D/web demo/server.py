from flask import Flask, request, jsonify
from flask_cors import CORS
from clip_embedding import get_text_embedding
from generate_blendshapes import generate_blendshapes
import os
import traceback

app = Flask(__name__)
# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/process_text', methods=['POST', 'OPTIONS'])
def process_text():
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        print("Received request to process text")
        data = request.get_json()
        print(f"Received data: {data}")
        
        text = data.get('text')
        
        if not text:
            print("No text provided in request")
            return jsonify({'error': 'No text provided'}), 400
            
        print(f"Processing text: {text}")
        
        # Get the embedding using our clip_embedding.py function
        embedding = get_text_embedding(text)
        
        # Generate blend shapes
        print("Generating blend shapes...")
        blend_shapes, controller_values = generate_blendshapes()
        
        # Verify files were created
        current_dir = os.getcwd()
        embedding_path = os.path.join(current_dir, 'text_embedding.npy')
        blendshapes_path = os.path.join(current_dir, 'generated_blendshapes.npy')
        controllers_path = os.path.join(current_dir, 'controller_values.npy')
        
        response_data = {
            'message': 'Text embedding and blend shapes processed successfully',
            'embedding_shape': embedding.shape,
            'blendshapes_shape': blend_shapes.shape,
            'controller_values': controller_values,
            'embedding_path': embedding_path,
            'blendshapes_path': blendshapes_path,
            'controllers_path': controllers_path
        }
        
        if os.path.exists(embedding_path):
            print(f"Successfully verified embedding file exists at: {embedding_path}")
            response_data['embedding_size'] = os.path.getsize(embedding_path)
            
        if os.path.exists(blendshapes_path):
            print(f"Successfully verified blend shapes file exists at: {blendshapes_path}")
            response_data['blendshapes_size'] = os.path.getsize(blendshapes_path)
            
        if os.path.exists(controllers_path):
            print(f"Successfully verified controller values file exists at: {controllers_path}")
            response_data['controllers_size'] = os.path.getsize(controllers_path)
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        error_response = jsonify({'error': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

if __name__ == '__main__':
    print(f"Server starting in directory: {os.getcwd()}")
    app.run(debug=True, host='0.0.0.0', port=5003) 