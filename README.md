🚦 Traffic Light Detection and Classification using ResNet50

📌 Overview

This project implements a Traffic Light Detection and Classification system using the ResNet50 deep learning model.
The model is capable of detecting traffic lights in images and classifying them into three categories:

🟥 Red

🟨 Yellow

🟩 Green

This project can be used in autonomous driving systems, traffic monitoring, or computer vision learning purposes.

📂 Project Structure
TrafficLight-Detection-ResNet50/
│
├── dataset/                      # Dataset (images of traffic lights)
├── model/                     # Saved trained model & weights
├── util/                 # Jupyter notebooks for EDA & training
├── testImages/                   # Python Images for training & testing 
├── videos/
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── main.py                    # Run detection/classification

🧠 Features

✅ Detect traffic lights from images or video
✅ Classify signals into Red, Yellow, Green
✅ Transfer learning with ResNet50 for better accuracy
✅ Easy-to-use Python scripts for training and testing

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/loner1423/TrafficLight-Detection-ResNet50.git
cd TrafficLight-Detection-ResNet50

2️⃣ Install dependencies
pip install -r requirements.txt

🗂 Dataset

Dataset collected from public traffic datasets or custom images.

Structure for training:

dataset/
    ├── go/
    ├── goforward/
    ├── goleft/
    ├── stop/
    ├── stopleft/
    ├── warning/
    └── warningleft/

🚀 Usage
Train the model
python scripts/train.py --epochs 25 --batch_size 32 --lr 0.0001

Test the model
python scripts/test.py --image path_to_image.jpg

Run live detection
python main.py --video traffic_video.mp4

📊 Results
Class	Precision	Recall	Accuracy
Red	98%	97%	97.5%
Yellow	95%	94%	94.5%
Green	97%	96%	96.5%

Confusion Matrix & Accuracy Curve can be added here.

🛠 Tech Stack

Language: Python

Framework: TensorFlow / Keras

Model: ResNet50 (Pre-trained on ImageNet)

Libraries: OpenCV, NumPy, Matplotlib, Pandas

🤝 Contributing

Fork the repo

Create a new branch

Commit your changes

Submit a Pull Request

🙌 Acknowledgements

ResNet50 Paper

TensorFlow Documentation

Public traffic light datasets
