Deepfake Detection System using Spatial Artifact Analysis

📌 Project Overview

This project is a high-accuracy Deepfake Detection system built using PyTorch. It utilizes a ResNet-50 architecture fine-tuned to detect microscopic spatial artifacts and inconsistencies commonly found in AI-generated facial media. The system achieved a validation accuracy of 99.26% during the fine-tuning phase.

This repository serves as Module 1 (Spatial Detector) of a multi-modal liveness detection framework designed for final-year engineering research.

📜 Publication

If you find this work useful in your research, please consider citing our published paper:

[DEEPFAKE SECURITY SUITE: A DEEP LEARNING-BASED SYSTEM 
FOR REAL-TIME DEEPFAKE DETECTION ] > [Dr. Ramya BN1, Ankitha G Kulkarni2, Dawan N Gowda3, J Manu4 ], [INTERNATIONAL JOURNAL OF PROGRESSIVE 
RESEARCH IN ENGINEERING MANAGEMENT 
AND SCIENCE (IJPREMS) ], [2025]

(https://www.ijprems.com/ijprems-paper/deepfake-security-suite-a-deep-learning-based-system-for-real-time-deepfake-detection)

🚀 Key Features

High Precision: Fine-tuned ResNet-50 architecture achieving >99% accuracy.

Hybrid Training: Implements an initial frozen-base training followed by an all-layer fine-tuning phase.

Web Interface: Integrated Flask application for real-time image and video analysis.

Robust Processing: Automated face detection and alignment using OpenCV Haar Cascades.

Resumable Training: Full checkpointing system to save/load training states including optimizer and scheduler states.

📂 Project Structure

deepfake-detection/
├── app/
│   ├── static/             # CSS and JS files
│   ├── templates/
│   │   └── index_pytorch.html
│   └── app_pytorch.py      # Flask Backend
├── models/
│   ├── architecture.py     # Model class definitions
│   └── .gitkeep            # Folder for .pth weights (ignored by git)
├── scripts/
│   ├── train_pytorch.py    # Combined Training & Fine-tuning script
│   └── extract_faces.py    # Preprocessing script
├── .gitignore              # Files to exclude from GitHub
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


🛠️ Installation & Setup

Clone the repository:

git clone [https://github.com/yourusername/deepfake-detection.git](https://github.com/yourusername/deepfake-detection.git)
cd deepfake-detection


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Model Weights:
Due to GitHub's file size limits, the trained model weights (best_model_pytorch_finetuned.pth) are not included.

[Link to Download Model Weights from Google Drive/OneDrive]

💻 Usage

Running the Web App

python app/app_pytorch.py


Open http://127.0.0.1:5000 in your browser to upload images or videos for testing.

Training

To restart or continue training:

python scripts/train_pytorch.py


📊 Training Performance

The model was trained on a dataset of ~127,000 images.

Initial Phase (Frozen Base): Reached plateau at ~91% accuracy.

Fine-Tuning Phase (All Layers): Reached peak validation accuracy of 99.26% at Epoch 25.

⚖️ License

This project is licensed under the MIT License.

🤝 Acknowledgments

PyTorch Team for the ResNet implementation.

OpenCV for the face detection modules.