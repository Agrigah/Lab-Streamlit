import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_cnn import prepare_image

base = Path(__file__).resolve().parent
model = tf.keras.models.load_model(base / 'CNN_fruits.h5', compile=False)
model.save(base / 'CNN_fruits.keras')
restored = tf.keras.models.load_model(base / 'CNN_fruits.keras', compile=False)
classes = json.loads((base / 'class_names.json').read_text(encoding='utf-8-sig'))
confusion = np.zeros((3, 3), dtype=int)
for actual, label in enumerate(classes):
    for path in sorted((base / 'fruits' / 'test' / label).iterdir()):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
            continue
        with Image.open(path) as image:
            batch = prepare_image(image)
        prediction = model(batch, training=False).numpy()
        np.testing.assert_allclose(prediction, restored(batch, training=False).numpy(), rtol=1e-5, atol=1e-6)
        confusion[actual, int(prediction.argmax())] += 1
result = {'source': 'Corrected CNN with ReLU convolution layers, trained locally', 'class_names': classes, 'test_images': int(confusion.sum()), 'accuracy': float(confusion.trace() / confusion.sum()), 'confusion_matrix_actual_rows_predicted_columns': confusion.tolist(), 'keras_export_matches_h5': True}
(base / 'model_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
from streamlit.testing.v1 import AppTest
app = AppTest.from_file(str(base / 'streamlit_cnn.py')).run(timeout=90)
assert not app.exception, app.exception
assert not app.error, app.error
print('Streamlit startup test passed')
