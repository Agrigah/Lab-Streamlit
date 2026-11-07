import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / 'natural_images_config.json'

@st.cache_resource
def load_assets(signature):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
    model = tf.keras.models.load_model(BASE_DIR / config['model_file'], compile=False)
    classes = json.loads((BASE_DIR / config['class_names_file']).read_text(encoding='utf-8-sig'))
    if not isinstance(classes, list) or len(classes) != model.output_shape[-1] or len(set(classes)) != len(classes):
        raise ValueError('Class names must match the model outputs in training order.')
    if tuple(model.input_shape[1:]) != (*config['image_size'], 3):
        raise ValueError('Image dimensions in the configuration do not match the model.')
    if config['preprocessing'] != 'mobilenet_v2':
        raise ValueError('Unsupported preprocessing configuration.')
    @tf.function(input_signature=[tf.TensorSpec([None, *config['image_size'], 3], tf.float32)])
    def predict(batch):
        return model(batch, training=False)
    return predict, classes, config


def prepare_image(image, image_size=(150, 150)):
    pixels = np.asarray(ImageOps.exif_transpose(image).convert('RGB'), dtype=np.float32)
    resized = tf.image.resize(pixels, image_size, method='bilinear')
    # This model has no internal Rescaling layer. Apply MobileNetV2 preprocessing once.
    return tf.expand_dims(tf.keras.applications.mobilenet_v2.preprocess_input(resized), 0)


def main():
    st.set_page_config(page_title='Natural image classifier', page_icon='📷')
    st.title('Natural image classifier')
    st.write('Classify an image using MobileNetV2.')
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
        paths = [CONFIG_PATH, BASE_DIR / config['model_file'], BASE_DIR / config['class_names_file']]
        signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        predict, classes, config = load_assets(signature)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f'Cannot load the natural-images model: {exc}')
        st.stop()
    st.write('Supported classes: ' + ', '.join(label.title() for label in classes) + '.')
    st.caption(f'MobileNetV2 · {config["image_size"][0]} × {config["image_size"][1]} RGB')
    with st.expander('Model details'):
        st.write('Model: ' + config['model_file'])
        st.write('Output order: ' + ', '.join(classes))
        st.code(str(BASE_DIR))
        if not config.get('training_metadata_confirmed', False):
            st.info('Class order follows the supplied natural-images notebooks. MobileNetV2 preprocessing is assumed; confirm both with the training notebook for this model.')
    uploaded = st.file_uploader('Upload an image', type=['png','jpg','jpeg','webp'])
    if uploaded is None:
        st.info('Upload an image to get a prediction.')
        return
    try:
        with Image.open(uploaded) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        st.image(image, caption='Uploaded image', width='stretch')
        with st.spinner('Classifying...'):
            probabilities = predict(prepare_image(image, tuple(config['image_size']))).numpy()[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError('The model returned invalid scores.')
    except (UnidentifiedImageError, OSError, ValueError, tf.errors.OpError) as exc:
        st.error(f'Could not process this image: {exc}')
        return
    winner = int(np.argmax(probabilities))
    st.success(f'Prediction: **{classes[winner].title()}**')
    st.metric('Model confidence', f'{probabilities[winner]:.2%}')
    st.subheader('Class scores')
    for index in np.argsort(probabilities)[::-1]:
        st.write(f'{classes[index].title()}: {probabilities[index]:.2%}')
    st.caption('The model always chooses from these eight classes. Confidence is not a guarantee of correctness.')

if __name__ == '__main__':
    main()
