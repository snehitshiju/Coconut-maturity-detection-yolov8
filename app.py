from flask import Flask, render_template, request, jsonify
import os
import uuid
from detect import run_detection

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PNG, JPG, JPEG allowed'}), 400

    # Save uploaded image with unique name
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = str(uuid.uuid4()) + '.' + ext
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    result_name = 'result_' + unique_name
    result_path = os.path.join(RESULT_FOLDER, result_name)

    file.save(upload_path)

    try:
        detections = run_detection(upload_path, result_path)

        # Not a coconut image
        if detections == 'no_coconut':
            return jsonify({'error': 'no_coconut'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'result_image': result_path.replace('\\', '/'),
        'detections': detections
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
