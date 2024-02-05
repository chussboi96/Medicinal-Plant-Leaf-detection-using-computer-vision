import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np

# Load the trained model
model_path = 'medicinal_leaf_classifier.h5'
model = tf.keras.models.load_model(model_path)

# Hardcoded class names
label_mapping = {
    0: 'Carissa Carandas (Karanda)', 1: 'Citrus Limon (Lemon)', 2: 'Ficus Auriculata (Roxburgh fig)',
    3: 'Ficus Religiosa (Peepal Tree)', 4: 'Hibiscus Rosa-sinensis',
    5: 'Jasminum (Jasmine)', 6: 'Mangifera Indica (Mango)',
    7: 'Mentha (Mint)', 8: 'Moringa Oleifera (Drumstick)',
    9: 'Muntingia Calabura (Jamaica Cherry-Gasagase)', 10: 'Murraya Koenigii (Curry)',
    11: 'Nerium Oleander (Oleander)', 12: 'Nyctanthes Arbor-tristis (Parijata)',
    13: 'Ocimum Tenuiflorum (Tulsi)', 14: 'Piper Betle (Betel)',
    15: 'Plectranthus Amboinicus (Mexican Mint)', 16: 'Pongamia Pinnata (Indian Beech)',
    17: 'Psidium Guajava (Guava)', 18: 'Punica Granatum (Pomegranate)', 19: 'Santalum Album (Sandalwood)',
    20: 'Syzygium Cumini (Jamun)', 21: 'Syzygium Jambos (Rose Apple)'
}

def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def predict_plant(img, label_mapping):
    preprocessed_image = preprocess_image(img)
    predictions = model.predict(preprocessed_image)

    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, "Unknown class")
    confidence = predictions[0][predicted_label_index]

    return predicted_label, confidence


st.title("Medicinal Leaf Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = image.load_img(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)
    st.write("Classifying...")

    predicted_label, confidence = predict_plant(img, label_mapping)
    st.write(f'Predicted class: {predicted_label}')
    st.write(f'Confidence: {confidence:.2f}')
