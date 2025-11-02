import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications import VGG16
import io

# --- CRITICAL CONFIGURATION (7-Class VGG16) ---
MODEL_PATH = 'facial_emotion_vgg16_7class.h5' # Must match the model you uploaded via LFS
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
IMG_SIZE = 48
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] 

st.set_page_config(
    page_title="VGG16 Emotion Recognition", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 1. MODEL LOADING (CRITICAL FIX APPLIED HERE) ---
@st.cache_resource
def load_resources():
    """
    Loads the model using custom_objects to correctly initialize the VGG16 layer,
    resolving the 'list' object has no attribute 'shape' error during deployment.
    """
    try:
        # Suppress TensorFlow warnings/messages during load
        tf.get_logger().setLevel('ERROR') 
        
        # Define the custom objects map to inform Keras how to handle the VGG16 base layer
        custom_objects = {"VGG16": VGG16}

        # Load the model using the custom_objects argument
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects=custom_objects, 
            compile=False
        ) 
        
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        
        if face_cascade.empty():
             st.error("Error: Face detector failed to load the XML file.")
             return None, None
             
        return model, face_cascade
    except Exception as e:
        st.error(f"Failed to load AI Model: {e}")
        st.warning(f"Check logs. Ensure '{MODEL_PATH}' was pushed with Git LFS.")
        return None, None

model, face_cascade = load_resources()

# --- 2. CORE PREDICTION LOGIC (VGG16 RGB INPUT) ---
def predict_emotion(image, model, face_cascade):
    
    img_array = np.array(image.convert('RGB')) 
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
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
        st.error("Application setup failed. Please check the deployment logs.")
        st.stop()
    
    st.info("The model is ready to classify 7 emotions. Focus the demo on Happy/Neutral for $\mathbf{80\%+}$ confidence.")
    
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
