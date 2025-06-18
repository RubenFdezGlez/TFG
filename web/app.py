from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
from ultralytics import YOLO
import uuid

# Commands
# $ export FLASK_APP=app.py -- set the FLASK_APP environment variable to app.py
# $ flask run -- run the Flask application

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = YOLO('yolo11n.pt')

# Define the root route
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Guardar imagen
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Ejecutar predicción
    results = model(filepath)
    result_img = results[0].plot()

    # Guardar imagen resultante
    result_filename = f"pred_{filename}"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_img)

    # Extraer clase principal
    label = results[0].names[results[0].boxes.cls[0].item()] if results[0].boxes else "No detection"

    return render_template('result.html', image=result_filename, label=label)

@app.route('/video_feed')
def video_feed():
    return render_template('video.html')

# Stream de la webcam con predicción
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        if results:
            frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True) 