from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from train_model import create_model 
import tensorflow as tf
from predict import predict_pollution_level

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model
model = None
try:
    model = tf.keras.models.load_model('waterpollution/project/pollution_level_model1.keras')
except:
    print("Model not found, please train the model first.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction (already returns text category)
            prediction_text = predict_pollution_level(model, filepath)
            return render_template('result.html', 
                                filename=filename, 
                                prediction=prediction_text)
    return render_template('upload.html')

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)