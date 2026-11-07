import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_assets(model_signature):
    model_path = BASE_DIR / 'CNN_fruits.keras'
    if not model_path.exists():
        model_path = BASE_DIR / 'CNN_fruits.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    classes = json.loads((BASE_DIR / 'class_names.json').read_text(encoding='utf-8-sig'))
    if model.output_shape[-1] != len(classes):
        raise ValueError('Model output and class names do not match. Retrain the model.')
    return model, classes


def prepare_image(image):
    # Training resizes with TensorFlow bilinear interpolation. The model rescales pixels.
    pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.float32)
    return tf.expand_dims(tf.image.resize(pixels, (32, 32)), axis=0)


def main():
    st.title('Fruit classification application')
    st.write('Upload an image to classify it as an apple, banana, or orange.')
    try:
        model_path = BASE_DIR / 'CNN_fruits.keras'
        if not model_path.exists():
            model_path = BASE_DIR / 'CNN_fruits.h5'
        model_signature = (str(model_path), model_path.stat().st_mtime_ns, (BASE_DIR / 'class_names.json').stat().st_mtime_ns)
        model, classes = load_assets(model_signature)
    except (OSError, ValueError) as exc:
        st.error(f'Cannot load the model: {exc}')
        st.info('Run .\\.venv\\Scripts\\python.exe train_model.py first.')
        st.stop()
    st.caption(f'Model: {model_path.name} | Input: 32 × 32 RGB | Classes: {", ".join(classes)}')
    with st.expander('App and model location'):
        st.code(str(BASE_DIR))
    uploaded_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is None:
        st.info('Please upload an image to classify.')
        return
    try:
        with Image.open(uploaded_file) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        st.image(image, caption='Uploaded image', width='stretch')
        predictions = model(prepare_image(image), training=False).numpy()[0]
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f'Could not process this image: {exc}')
        return
    winner = int(np.argmax(predictions))
    st.subheader('Prediction results')
    st.success(f'Predicted fruit: **{classes[winner].title()}**')
    st.write(f'Model confidence: {predictions[winner]:.2%}')
    st.subheader('Class predictions')
    for label, probability in zip(classes, predictions):
        st.write(f'{label.title()}: {probability:.2%}')
    st.caption('This model only distinguishes these three fruits; other images will still receive one of these labels.')


if __name__ == '__main__':
    main()


