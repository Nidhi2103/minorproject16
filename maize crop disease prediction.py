import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder

# Load and preprocess image data
def load_and_preprocess_image(image_path, image_height, image_width):
    img = load_img(image_path, target_size=(image_height, image_width))
    img_array = img_to_array(img)
    return np.expand_dims(img_array / 255.0, axis=0)  # Normalize and add batch dimension

# Create textual symptom dictionary
def create_textual_symptom_dictionary():
    symptom_dict = {
        'Common Rust': "Symptoms include small, circular to oval, reddish-brown to tan lesions that form mainly on the upper surface of leaves.",
        'Blight': "Symptoms include cigar-shaped lesions that are initially gray-green but turn tan with age, and may have wavy margins.",
        'Gray Leaf Spot': "Symptoms include elongated lesions appears on lower leaves,lesions increase long with yelloworange borders and tanned center",
        'Healthy':"A healthy maize leaf typically has a vibrant green color, leaves are elongated and lance-shaped,no yellowing or browning."
        # Add more diseases and their symptoms as needed
    }
    return symptom_dict

# Define parameters
image_height, image_width = 224, 224  # ResNet50 input size
max_text_length = 100  # Define max text length
embedding_dim = 100  # Define embedding dimension for text
lstm_units = 128  # Define number of LSTM units

# Define the ResNet-50 model for image processing
image_input = Input(shape=(image_height, image_width, 3))
resnet_model = ResNet50(weights='imagenet', include_top=False)
for layer in resnet_model.layers:
    layer.trainable = False
image_features = resnet_model(image_input)

# Define the LSTM model for text processing
text_input = Input(shape=(max_text_length,))
text_embedding = Embedding(input_dim=1, output_dim=embedding_dim)(text_input)  # Dummy embedding layer
lstm_output = LSTM(units=lstm_units)(text_embedding)

# Combine image and text features
image_features_flat = Flatten()(image_features)
combined_features = concatenate([image_features_flat, lstm_output])

# Add dense layers for classification
dense1 = Dense(256, activation='relu')(combined_features)
output = Dense(len(create_textual_symptom_dictionary()), activation='softmax')(dense1)

# Create the model
model = Model(inputs=[image_input, text_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load pre-trained weights (if available)
# model.load_weights("path/to/pretrained/weights.h5")

# Input image and text for prediction
input_image_path = "/content/corn-leaves (1).jpg"
input_text = "Vibrant Green Color,no yellowing or browning."
# Load and preprocess input image
input_image_data = load_and_preprocess_image(input_image_path, image_height, image_width)

# Dummy text input
input_text_data = np.zeros((1, max_text_length))  # Dummy text input

# Make prediction
prediction = model.predict([input_image_data, input_text_data])

# Decode prediction
predicted_class_index = np.argmax(prediction)
predicted_class = list(create_textual_symptom_dictionary().keys())[predicted_class_index]
predicted_symptom = list(create_textual_symptom_dictionary().values())[predicted_class_index]

print("Predicted Disease Class:", predicted_class)
print("Predicted Disease Symptoms:", predicted_symptom)
