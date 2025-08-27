Got it — your dataset has **7 classes** instead of 3.
Here’s an updated **GitHub README** with your **exact dataset structure and classes**:

---

# 🚦 Traffic Light Detection and Classification using ResNet50

![Traffic Light](https://img.shields.io/badge/ML-ResNet50-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📌 Overview

This project implements a **Traffic Light Detection and Classification system** using the **ResNet50 deep learning model**.
The model is trained to detect and classify traffic lights into the following **7 classes**:

| Class ID | Traffic Light Action |
| -------- | -------------------- |
| **1**    | 🟢 Go                |
| **2**    | ⬆ Go Forward         |
| **3**    | ↖ Go Left            |
| **4**    | ⛔ Stop               |
| **5**    | ⛔↖ Stop & Left       |
| **6**    | ⚠ Warning            |
| **7**    | ⚠↖ Warning & Left    |

---

## 📂 Project Structure

```
TrafficLight-Detection-ResNet50/
│
├── data/                      # Dataset
│   ├── train/                 # Training dataset
│   │   ├── go/
│   │   ├── goforward/
│   │   ├── goleft/
│   │   ├── stop/
│   │   ├── stopleft/
│   │   ├── warning/
│   │   └── warningleft/
│   └── test/                  # Testing dataset
│       ├── go/
│       ├── goforward/
│       ├── goleft/
│       ├── stop/
│       ├── stopleft/
│       ├── warning/
│       └── warningleft/
│
├── model/                     # Saved trained model & weights
├── notebooks/                 # Jupyter notebooks for EDA & training
├── scripts/                   # Python scripts for training & testing
│   ├── train.py               # Script to train the model
│   ├── test.py                # Script to test predictions
│   └── utils.py               # Helper functions
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── main.py                    # Run detection/classification
```

---

## 🧠 Features

✅ Detect traffic lights from **images, videos, or live camera**
✅ Classify into **7 specific classes**
✅ Uses **transfer learning with ResNet50** for high accuracy
✅ Easy training and testing scripts

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/TrafficLight-Detection-ResNet50.git
cd TrafficLight-Detection-ResNet50
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🗂 Dataset

Organize your dataset in this structure before training:

```
data/
├── train/
│   ├── go/
│   ├── goforward/
│   ├── goleft/
│   ├── stop/
│   ├── stopleft/
│   ├── warning/
│   └── warningleft/
└── test/
    ├── go/
    ├── goforward/
    ├── goleft/
    ├── stop/
    ├── stopleft/
    ├── warning/
    └── warningleft/
```

---

## 🚀 Usage

### **Train the model**

```bash
python scripts/train.py --epochs 30 --batch_size 32 --lr 0.0001
```

### **Test the model**

```bash
python scripts/test.py --image path_to_image.jpg
```

### **Run live detection**

```bash
python main.py --video traffic_video.mp4
```

---

## 📊 Results

| Class        | Precision | Recall | Accuracy |
| ------------ | --------- | ------ | -------- |
| Go           | 98%       | 97%    | 97.5%    |
| Go Forward   | 96%       | 95%    | 95.5%    |
| Go Left      | 95%       | 94%    | 94.5%    |
| Stop         | 99%       | 98%    | 98.5%    |
| Stop Left    | 97%       | 96%    | 96.5%    |
| Warning      | 94%       | 93%    | 93.5%    |
| Warning Left | 93%       | 92%    | 92.5%    |

*(Update with your actual results after training)*

---

## 🛠 Tech Stack

* **Language**: Python
* **Framework**: TensorFlow / Keras
* **Model**: ResNet50 (Pre-trained on ImageNet)
* **Libraries**: OpenCV, NumPy, Matplotlib, Pandas

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch
3. Commit your changes
4. Submit a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [ResNet50 Paper](https://arxiv.org/abs/1512.03385)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* Public traffic light datasets

---

Would you like me to **add example code snippets** for training and prediction to this README?
