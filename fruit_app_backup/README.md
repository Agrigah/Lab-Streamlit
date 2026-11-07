# Fruit classifier

Uses the corrected CNN trained locally with ReLU convolution layers and early stopping. Both model formats are saved. No retraining is needed to launch the app.

From PowerShell in this folder:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m streamlit run streamlit_cnn.py

Or double-click run_app.bat after dependencies are installed.

The app loads CNN_fruits.keras when present, otherwise CNN_fruits.h5.
Class order is apple, banana, orange, from the notebook's alphabetically sorted training folders.
Input is RGB, resized to 32 x 32 using TensorFlow bilinear interpolation. Pixel values stay in the 0-255 range because the saved model contains a Rescaling layer.

Optional retraining (overwrites the model files):

    .\.venv\Scripts\python.exe train_model.py --epochs 30

Validate the current H5 model, export Keras format, evaluate test images, and test app startup:

    .\.venv\Scripts\python.exe verify_model.py

The original app is preserved as streamlit_cnn.original.py. The source notebook and local fruits dataset are included.

Fixes: normalize only inside the model, use saved alphabetical class labels, add ReLU to convolution layers, and invalidate the Streamlit model cache when model files change. Previous files are in backup_before_cnn_fix. Original bug reproduction is recorded in original_bug_diagnosis.json.

Stop any old Streamlit server with Ctrl+C, then run .\run_app.bat from this folder. Check the App and model location expander in the browser to confirm you launched this copy.
