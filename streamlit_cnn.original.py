import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
    model=tf.keras.models.load_model('CNN_fruits.h5')
    return model

model = load_model()

classes=['Banana','Orange','Apple']
st.title('Fruit classification Application')
st.write("Upload an Image of a fruit to identify it it's a **banana**, **orange** or **apple**.")

uploaded_file = st.file_uploader("Upload in Image...",type = ["png","jpg","jpeg"])

if uploaded_file is not None:
    image=Image.open(uploaded_file).convert("RGB")
    st.image(image,caption = "Uploaded Image",use_column_width=True)

    img=image.resize((32,32))
    img_array=np.array(img)/255.0
    img_array=np.expand_dims(img_array,axis=0)

    predictions = model.predict(img_array)
    predicted_class=classes[np.argmax(predictions)]
    confidence=np.max(predictions)*100

    st.subheader("Prediction Results")
    st.success(f"**Predicted fruit:** {predicted_class}")
    st.write(f"confidence score: {confidence:.2f} %")

    st.write("### Class predictions")
    for i,label in enumerate(classes):
        st.success(f"{label}: {predictions[0][i]*100:.2f}%")
else:
    st.info("Please upload an image to classify .")
