# Coconut-maturity-detection-yolov8🥥 CocoVision: Coconut Maturity Detection System (YOLOv8 + Flask)

CocoVision is an AI-powered web application that detects coconut maturity stages using a YOLOv8 deep learning model. The system classifies coconuts into Premature, Potential, and Mature stages and provides harvest timeline recommendations through an interactive web interface.

Designed for precision agriculture, CocoVision enables fast, reliable, and explainable maturity detection from images.

🚀 Features
Detects coconut maturity using image input
Classifies into:
Premature
Potential
Mature
Generates confidence scores
Provides harvest readiness timeline prediction
Displays annotated bounding boxes
Supports multi-object detection
Includes Grad-CAM explainable AI visualization
Lightweight YOLOv8 model (6.2 MB)
Real-time detection via Flask web interface
🧠 Model Information
Parameter	Value
Model	YOLOv8
Classes	3
Model Size	6.2 MB
Parameters	~3 Million
CPU Inference	~2.1 ms
GPU Inference	<1 ms

⚙️ Installation
Step 1: Clone Repository
git clone https://github.com/yourusername/CocoVision.git
cd CocoVision
Step 2: Install Dependencies
pip install -r requirements.txt

If requirements file is unavailable:

pip install flask ultralytics opencv-python numpy
Step 3: Run Application
python app.py

Open browser:

http://localhost:5000

Upload coconut image and view predictions.

🧪 Model Training (Optional)

To retrain model using your dataset:

yolo detect train data=data.yaml model=yolov8n.pt epochs=50

After training completes:

runs/detect/train/weights/best.pt

Copy this file into:

runs/detect/coconut_model/weights/
📊 Dataset Format

Dataset follows YOLO format:

train/
valid/
test/

Example data.yaml:

train: dataset/train/images
val: dataset/valid/images
test: dataset/test/images

nc: 3
names:
- Mature
- Potential
- Premature
🖥️ System Workflow
User uploads image
        ↓
Flask server receives request
        ↓
detect.py loads YOLOv8 model
        ↓
Model predicts maturity stage
        ↓
Bounding boxes generated
        ↓
Timeline estimated
        ↓
Results displayed in browser
🔍 Explainable AI Support

Grad-CAM visualization highlights image regions influencing predictions:

husk color gradients
texture variations
fruit curvature

Improves transparency and trust in predictions.

🌾 Applications
Smart farming assistance
Automated harvest scheduling
Plantation monitoring
Drone-based crop inspection (future scope)
IoT agriculture systems
🔮 Future Improvements
Mobile application deployment
Drone-based maturity monitoring
Cloud analytics dashboard
Yield prediction integration
Disease detection module
Edge-device deployment (Jetson / Raspberry Pi)
