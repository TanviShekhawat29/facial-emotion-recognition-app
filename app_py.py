import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import io

# --- CRITICAL CONFIGURATION (7-Class VGG16) ---
MODEL_PATH = 'facial_emotion_vgg16_7class.h5' 
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
IMG_SIZE = 48
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 
NUM_CLASSES = 7

st.set_page_config(
    page_title="VGG16 Emotion Recognition", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- MODEL ARCHITECTURE DEFINITION (REPLICATED FROM TRAINING SCRIPT) ---
def build_vgg16_transfer_model(input_shape, num_classes):
    """Defines the exact model architecture to load the weights correctly."""
    # 1. Load VGG16 Base
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    
    # Freeze the weights of the VGG16 layers
    conv_base.trainable = False 
    
    # 2. Build Classification Head
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') 
    ])
    
    # Do not compile, as we only load weights.
    return model

# --- 1. MODEL LOADING (FINAL ROBUST FIX) ---
@st.cache_resource
def load_resources():
    """Builds the architecture and loads weights manually to avoid Keras serialization errors."""
    try:
        tf.get_logger().setLevel('ERROR') 
        
        # 1. Build the empty model architecture
        input_shape = (IMG_SIZE, IMG_SIZE, 3) 
        model = build_vgg16_transfer_model(input_shape, NUM_CLASSES)

        # 2. Load the trained weights onto the matching structure
        # Use load_weights instead of load_model to bypass serialization issues
        model.load_weights(MODEL_PATH)
        
        # Load Haar Cascade
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        
        if face_cascade.empty():
             st.error("Error: Face detector failed to load the XML file.")
             return None, None
             
        return model, face_cascade
    except Exception as e:
        st.error(f"Failed to load AI Model architecture or weights: {e}")
        st.warning(f"Check logs. Ensure '{MODEL_PATH}' was pushed with Git LFS.")
        return None, None

model, face_cascade = load_resources()

# --- 2. CORE PREDICTION LOGIC (VGG16 RGB INPUT) ---
def predict_emotion(image, model, face_cascade):
    
    img_array = np.array(image.convert('RGB')) 
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return img_array, 0

    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y + h, x:x + w]
        cropped_img_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        
        # CRUCIAL: Convert 1-channel grayscale face patch to 3-channel RGB for VGG16
        cropped_img_rgb = cv2.cvtColor(cropped_img_gray, cv2.COLOR_GRAY2BGR) 
        
        # Normalize and reshape (1, 48, 48, 3)
        processed_input = cropped_img_rgb.astype('float32') / 255.0
        processed_input = np.expand_dims(processed_input, axis=0) 
        
        # Make Prediction
        predictions = model.predict(processed_input, verbose=0)[0]
        
        emotion_index = np.argmax(predictions)
        emotion_label = EMOTIONS[emotion_index]
        confidence = predictions[emotion_index] * 100
        
        # Draw Results
        color = (255, 165, 0) # Orange/Yellow
        
        cv2.rectangle(img_array, (x, y), (x + w, y + h), color, 2)
        text = f"{emotion_label}: {confidence:.1f}%"
        cv2.putText(img_array, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return img_array, len(faces)

# --- 3. STREAMLIT UI ---
def main():
    st.title("🌟 7-Class Facial Emotion Recognition (VGG16)")
    st.subheader("B.Tech AI/DL Project: Transfer Learning Approach")
    st.write("---")

    if model is None:
        st.stop()
    
    st.info("The model is ready to classify 7 emotions. Demo success should focus on **Happy** and **Neutral** for $\mathbf{80\%+}$ confidence.")
    
    uploaded_file = st.file_uploader(
        "Upload an image file (.jpg, .jpeg, .png) containing faces", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Uploaded Image**")
                st.image(image, use_column_width=True)
            
            if st.button('Analyze Emotion', use_container_width=True):
                with st.spinner('Running VGG16 Transfer Learning Model...'):
                    result_img_array, face_count = predict_emotion(image, model, face_cascade)
                
                result_img = Image.fromarray(result_img_array)
                
                with col2:
                    st.markdown("**Emotion Analysis**")
                    if face_count > 0:
                        st.image(result_img, caption=f'Detected {face_count} face(s)', use_column_width=True)
                        st.success("Analysis Complete!")
                    else:
                        st.warning("No faces detected in the image.")
                        st.image(image, use_column_width=True)
                        
        except Exception as e:
            st.exception(e)
            st.error("An unexpected error occurred during processing.")

if __name__ == "__main__":
    main()
