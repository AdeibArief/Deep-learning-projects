import numpy as np
from tensorflow.keras import models
from PIL import Image
import tensorflow as tf
import streamlit as st

model=models.load_model('model.h5')

def predict_digit(image):
    image=image.resize((28,28))
    image=image.convert('L')
    image_array=np.array(image)
    image_array=image_array/255
    image_array=np.reshape(image_array,(1,28,28,1))
    prediction=model.predict(image_array)

    return np.argmax(prediction)

def main():
    st.title('Handwritten Digits Classification')
    st.sidebar.title('Uploaded image prediction')

    uploaded_image=st.file_uploader('choose a digit image',type=['jpeg','jpg','png'])

    if uploaded_image is not None:
        image=Image.open(uploaded_image)

        st.image(image,caption=f'uploaded image')
        prediction=predict_digit(image)
        st.sidebar.header('Prediction')
        st.sidebar.write(f'The digit in the image is {prediction}')

if __name__=="__main__":
    main()




























