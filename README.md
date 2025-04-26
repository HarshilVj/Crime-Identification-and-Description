# Crime-Detection-and-Description
🕵️‍♂️ Crime Detection and Description Using BiLSTM, MediaPipe, and InternVideo v2.5
This project focuses on automatic crime detection and description from video footage.
It combines a BiLSTM-based classifier (trained on pose landmarks extracted via MediaPipe Holistic) with a pretrained multimodal LLM (InternVideo v2.5) to generate natural language descriptions of detected activities.
We focus on distinguishing between two different types of crimes which were generating an extraordinary accuracy of about 98% as compared with multi-class classification accuracy (around 85%) , that are:
Shoplifting
Vandalism
The system aims to provide both a predicted crime label and a human-readable description for better interpretability in real-world surveillance and security applications majorly in public place like supermarkets, grocery stores, etc., where there are two primarily crimes shoplifting and vandalism.

The system processes video inputs and outputs:
- Predicted crime class
- A detailed, human-readable description of the event

---


## 🚀 Technologies Used

| Component                  | Technology                         |
|-----------------------------|------------------------------------|
| Human Pose Estimation       | MediaPipe Holistic                 |
| Video Processing            | OpenCV, Decord, ImageIO            |
| Crime Classification Model  | BiLSTM + Attention (TensorFlow)    |
| Video Understanding         | InternVideo v2.5 (via HuggingFace) |
| Frameworks                  | PyTorch, TensorFlow, Hugging Face  |

---

## ⚙️ Setup Instructions

### 1. Clone This Repository

git clone https://github.com/HarshilVj/Crime-Detection-and-Description.git

### 2. Install Required Packages
Install all the necessary dependencies by running:
pip install torch torchvision torchaudio
pip install tensorflow==2.12.0
pip install mediapipe opencv-python
pip install decord av imageio
pip install transformers==4.40.1 timm accelerate
pip install flash-attn --no-build-isolation  # (Optional for speed boost)

###3. Add Trained Models
Place the trained crime detection model at:
/content/crime_detection_model_2class.h5
Make sure you have a Hugging Face token (free account) for loading InternVideo. Set it like this in the code:
os.environ["HF_TOKEN"] = "your_huggingface_token"
📋 How It Works
1. Pose Landmark Extraction
The system uses MediaPipe Holistic to detect:

  Body pose (33 points)
  Left and right hand landmarks (21 points each)
These are stacked into a 225-dimensional feature vector per frame.

2. Crime Prediction
A custom-built BiLSTM + Attention model processes 100 frames and predicts the crime type.

3. Event Description
InternVideo generates a detailed description tailored to the predicted crime, leveraging a series of frames from the video.

4. Visualization
Frames are shown with the predicted crime class overlaid for visual confirmation.

▶️ How to Run the Project
Make sure all dependencies and weights are set. Then simply run:
python your_script_name.py (if running locally on your pc) or may be run directly as we did in google colab.
You will see:

🔹 Predicted Crime Class (example: Shoplifting)

📝 Description (example: "The individual discreetly hides merchandise into their bag without making a payment.")

Frames will be displayed with real-time overlays.
![WhatsApp Image 2025-04-26 at 19 39 33_b91b0b93](https://github.com/user-attachments/assets/cf9df037-b3fe-4a4d-9e92-63052074cadb)


📈 Model Performance

Metric	Value
Classification Accuracy	98%
Description Quality	Human-like
The model has been trained on curated datasets and optimized using multiple rounds of augmentation, making it highly reliable even for challenging videos.

📚 References and Resources
MediaPipe Holistic Documentation

InternVideo Research Paper

PyTorch Official Documentation

TensorFlow Official Documentation

Decord GitHub Repository

🚀 Future Scope
Expand to detect 5+ different crime types

Improve real-time streaming support (CCTV integration)

Fine-tune InternVideo specifically for surveillance footage

(Contributor:ShreyanshAgarwal17)
