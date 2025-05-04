import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
import json
import os

# Define the mapping between blend shape indices and face controllers
BLENDSHAPE_MAPPING = {
    'browDownLeft': 'browDown_L',
    'browDownRight': 'browDown_R',
    'browInnerUp': 'browinnerUp',
    'browOuterUpLeft': 'browOuterUp_L',
    'browOuterUpRight': 'browOuterUp_R',
    'cheekPuff': 'cheekPuff',
    'cheekSquintLeft': 'cheekSquint_L',
    'cheekSquintRight': 'cheekSquint_R',
    'eyeBlinkLeft': 'eyeBlink_L',
    'eyeBlinkRight': 'eyeBlink_R',
    'eyeLookDownLeft': 'eyeLookDown_L',
    'eyeLookDownRight': 'eyeLookDown_R',
    'eyeLookInLeft': 'eyeLookin_L',
    'eyeLookInRight': 'eyeLookIn_R',
    'eyeLookOutLeft': 'eyeLookOut_L',
    'eyeLookOutRight': 'eyeLookOut_R',
    'eyeLookUpLeft': 'eyeLookUp_L',
    'eyeLookUpRight': 'eyeLookUp_R',
    'eyeSquintLeft': 'eyeSquint_L',
    'eyeSquintRight': 'eyeSquint_R',
    'eyeWideLeft': 'eyeWide_L',
    'eyeWideRight': 'eyeWide_R',
    'jawForward': 'jawForward',
    'jawLeft': 'jawLeft',
    'jawOpen': 'jawOpen',
    'jawRight': 'jawRight',
    'mouthClose': 'mouthClose',
    'mouthDimpleLeft': 'mouthDimple_L',
    'mouthDimpleRight': 'mouthDimple_R',
    'mouthFrownLeft': 'mouthFrown_L',
    'mouthFrownRight': 'mouthFrown_R',
    'mouthFunnel': 'mouthFunnel',
    'mouthLeft': 'mouthLeft',
    'mouthLowerDownLeft': 'mouthLowerDown_L',
    'mouthLowerDownRight': 'mouthLowerDown_R',
    'mouthPressLeft': 'mouthPress_L',
    'mouthPressRight': 'mouthPress_R',
    'mouthPucker': 'mouthPucker',
    'mouthRight': 'mouthRight',
    'mouthRollLower': 'mouthRollLower',
    'mouthRollUpper': 'mouthRollUpper',
    'mouthShrugLower': 'mouthShrugLower',
    'mouthShrugUpper': 'mouthShrugUpper',
    'mouthSmileLeft': 'mouthSmile_L',
    'mouthSmileRight': 'mouthSmile_R',
    'mouthStretchLeft': 'mouthStretch_L',
    'mouthStretchRight': 'mouthStretch_R',
    'mouthUpperUpLeft': 'mouthUpperUp_L',
    'mouthUpperUpRight': 'mouthUpperUp_R',
    'noseSneerLeft': 'noseSneer_L',
    'noseSneerRight': 'noseSneer_R'
}

def create_model():
    model = Sequential([
        Input(shape=(512,), name='dense_3_input'),
        Dense(256, activation='relu', name='dense_3'),
        Dense(128, activation='relu', name='dense_4'),
        Dense(51, activation='sigmoid', name='dense_5')
    ])
    return model

def map_blendshapes_to_controllers(blend_shapes):
    """Map the generated blend shapes array to face controllers"""
    
    # Create a dictionary mapping controller names to their values
    controller_values = {}
    for i, (bs_name, controller_name) in enumerate(BLENDSHAPE_MAPPING.items()):
        controller_values[controller_name] = float(blend_shapes[i])
    
    return controller_values

def generate_blendshapes():
    # Load the text embedding
    print("Loading text embedding...")
    text_embedding = np.load('text_embedding.npy')
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Check if model files exist
    h5_path = 'bs_gen_model.h5'
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Model weights file not found at {h5_path}")
    
    # Create and load the model
    print("Loading BSGenModel...")
    try:
        # Create the model with the correct architecture
        model = create_model()
        
        # Load the weights from H5
        model.load_weights(h5_path)
        print("Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}. Please make sure the model files are valid.")
    
    # Generate blend shapes
    print("Generating blend shapes...")
    blend_shapes = model.predict(text_embedding)
    print(f"Generated blend shapes shape: {blend_shapes.shape}")
    blend_shapes = blend_shapes.flatten()
    bs_values = {}
    for i, (bs_name, controller_name) in enumerate(BLENDSHAPE_MAPPING.items()):
        bs_values[bs_name] = float(blend_shapes[i])

    # Map blend shapes to controllers
    controller_values = map_blendshapes_to_controllers(blend_shapes)
    
    # Save both the raw blend shapes and the mapped controller values
    with open('generated_blendshapes.json', 'w') as f:
        json.dump(bs_values, f)
    with open('controller_values.json', 'w') as f:
        json.dump(controller_values, f)
    np.save('generated_blendshapes.npy', blend_shapes)
    np.save('controller_values.npy', controller_values.values)
    print("Saved blend shapes to generated_blendshapes.npy")
    print("Saved controller values to controller_values.npy")
    
    return blend_shapes, controller_values

if __name__ == "__main__":
    blend_shapes, controller_values = generate_blendshapes()
    print("\nController values:")
    for controller, value in controller_values.items():
        print(f"{controller}: {value:.4f}") 