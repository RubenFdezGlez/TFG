# Flask application for YOLOv11 object detection + Yolov11 dog classification

# Import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
from ultralytics import YOLO
import uuid
import torch

# Start the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_det = YOLO('./models/yolov11n_det.pt').to(device)
model_cls = YOLO('./models/yolov11n_cls.pt').to(device)

# Define the root route
@app.route("/")
def index():
    return render_template("index.html")

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict using the models
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    det_results = model_det.predict(img_rgb, device=device, conf=0.5)
    cls_results = model_cls.predict(img, device=device)

    # Classify the detected dog breeds
    label = "No detection"
    if cls_results:
        label = cls_results[0].names[cls_results[0].probs.top1].split('-')[2]  # Extract the dog breed from the class name
        label = label.replace('_', ' ').capitalize()  # Format the label

    for box in det_results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the detected dog
        crop = img_rgb[y1:y2, x1:x2]

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
    result_img = img

    # Save the result image
    result_filename = f"pred_{filename}"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, result_img)

    return render_template('result.html', image=result_filename, label=label)

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return render_template('video.html')

# Webcam video streaming
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det_results = model_det.predict(img, device=device)
        cls_results = model_cls.predict(frame, device=device)

        # Classify the detected dog breeds
        label = "No detection"
        if cls_results:
            label = cls_results[0].names[cls_results[0].probs.top1].split('-')[2]  # Extract the dog breed from the class name
            label = label.replace('_', ' ').capitalize()  # Format the label

        for box in det_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the detected dog
            crop = img[y1:y2, x1:x2]

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 4)
        frame = img

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for streaming video
@app.route('/stream')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True) 