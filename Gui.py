import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from tensorflow.keras.models import load_model

model = load_model('model.h5')

st.write('# Handwritten Digit Recognition using a Neural Network')
st.divider()
st.write('Model: sequential')

#The Model
model_info = {
    'Layer (type)': ['flatten (Flatten)','dense (Dense)','dense_1 (Dense)', 'dense_2 (Dense)'],
    'Output Shape': ['(None, 784)','(None, 128)','(None, 64)','(None, 10)'],
    'Param #': ['0','100,480','8,286','650']
}

model_info_df = pd.DataFrame(model_info)

st.table(model_info_df)

st.divider()

#Results
st.write("Results")

with open('evaluation_metrics.pkl', 'rb') as file:
    metrics = pickle.load(file)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.metric("Validation Accuracy", f"{metrics['val_accuracy']*100:.2f}%")
    graph1 = Image.open('accuracy.png')
    st.image(graph1)

with col2:
    st.metric("Validation Loss", f"{metrics['val_loss']:.4f}")
    graph2 = Image.open('lost.png')
    st.image(graph2)

st.divider()

#Interactive Section
st.write('Demo')


test_data = pd.read_csv("test.csv")
X_test = test_data.values / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

num_of_rows = len(test_data)


if 'current_idx' not in st.session_state: #st.session_state is a dictionary and 'current_idx' is a key
    st.session_state.current_idx = random.randint(0, num_of_rows-1)

def display_random_image():
    image = X_test[st.session_state.current_idx].reshape(28,28)
    st.image(image, width=280)

display_random_image()

if st.button('Show a new image'):
    st.session_state.current_idx = random.randint(0, num_of_rows - 1)
    st.rerun()


if st.button('Predict'):

    image = X_test[st.session_state.current_idx:st.session_state.current_idx+1]
    prediction = model.predict(image, verbose = 0) #verbose 0 so you wont see the models processes in the console
    predicted_digit = np.argmax(prediction[0]) #np.argmax gets the index of the max element. prediction[0] b/c predict return 2D array
    confidence = prediction[0][predicted_digit] * 100
    st.write('Model believes this is a ' + str(predicted_digit) + ' with ' + str(confidence) + "% confidence.")




