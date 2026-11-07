from pathlib import Path
import json,time
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_natural_images import prepare_image

p=Path(__file__).resolve().parent
source=tf.keras.models.load_model(p/'mobilenetv2_natural_images.source.keras',compile=False)
# Loading without compilation discards optimizer state; saving preserves all learned weights.
source.save(p/'mobilenetv2_natural_images.keras')
restored=tf.keras.models.load_model(p/'mobilenetv2_natural_images.keras',compile=False)
classes=json.loads((p/'natural_images_class_names.json').read_text())
assert source.input_shape==(None,150,150,3)
assert source.output_shape==(None,len(classes))
for value,expected in [(0,-1.),(255,1.)]:
    batch=prepare_image(Image.new('RGB',(40,60),(value,value,value)))
    np.testing.assert_allclose(batch.numpy(),expected,atol=1e-6)
samples=[]
for name in ['cat.jpeg','dog.png']:
    f=p.parent/name
    if f.exists():samples.append(f)
samples+=list((p/'fruits/test/apple').glob('*.jpg'))[:1]
results=[]
for f in samples:
    with Image.open(f) as im:batch=prepare_image(im)
    expected=source(batch,training=False).numpy()
    actual=restored(batch,training=False).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    assert np.isfinite(actual).all() and actual.shape==(1,8)
    np.testing.assert_allclose(actual.sum(),1.,atol=1e-5)
    results.append({'image':str(f),'prediction':classes[int(actual.argmax())],'score':float(actual.max())})
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(p/'streamlit_natural_images.py')).run(timeout=120)
assert not app.exception,app.exception
assert not app.error,app.error
report={'source_bytes':(p/'mobilenetv2_natural_images.source.keras').stat().st_size,'inference_model_bytes':(p/'mobilenetv2_natural_images.keras').stat().st_size,'input_shape':list(source.input_shape),'output_classes':classes,'export_prediction_equivalence':True,'preprocessing_range_checks_passed':True,'streamlit_startup_passed':True,'sample_predictions':results,'accuracy':None,'limitation':'No matching labeled held-out dataset or MobileNetV2 training metadata supplied; sample predictions are smoke tests, not an accuracy measurement.'}
(p/'natural_images_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
