# 🦷 AI-Based Tooth Reconstruction

An AI-powered application that reconstructs damaged or missing tooth structures from images using deep learning. This project demonstrates how image inpainting techniques can be applied in dentistry to assist with visualization and treatment planning.

---

## 🚀 Demo

Live Demo (Hugging Face): [Add your link here]

---

## 📌 Overview

Dental injuries and decay often result in partial or missing tooth structures. Manual reconstruction requires expertise and time. This project aims to automate the reconstruction process using deep learning.

The system takes an input image of a damaged tooth and generates a reconstructed version by predicting the missing or damaged parts.

---

## 🧠 Features

- Upload tooth/dental image  
- Detect missing or damaged regions  
- Reconstruct tooth using AI model  
- Real-time prediction  
- Simple web interface using Gradio  

---

## 🏗️ Model Architecture

Input Image → Preprocessing → Deep Learning Model → Reconstructed Output

Model Used:
- U-Net / CNN-based Image Inpainting Model  
- Trained using original images and artificially masked images  

---

## 📊 Dataset

Due to limited availability of dental reconstruction datasets, this project uses:

- High-quality tooth images  
- Artificial masking technique:
  - Masked images → Input  
  - Original images → Ground truth  

This helps the model learn reconstruction effectively.

---

## ⚙️ Tech Stack

- Python  
- TensorFlow / PyTorch  
- OpenCV  
- NumPy  
- Gradio  
- Hugging Face Spaces  

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/tooth-reconstruction.git
cd tooth-reconstruction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
python app.py
```

Open in browser:
```
http://localhost:7860
```

---

## 🤖 Training the Model

Run training script:

```bash
python train.py
```

Steps:
- Load dataset  
- Apply masking to images  
- Train U-Net model  
- Save trained weights  

---

## 📁 Project Structure

```
tooth-reconstruction/
│
├── app.py              # Gradio UI
├── train.py            # Model training script
├── model/              # Saved model
├── data/               # Dataset
├── utils/              # Helper functions
├── requirements.txt
└── README.md
```

---

## 🌐 Deployment (Hugging Face)

1. Create a new Space on Hugging Face  
2. Upload all project files  
3. Add requirements.txt  
4. Set app.py as entry point  
5. Deploy 🚀  

---

## ⚠️ Limitations

- Depends on dataset quality  
- Works best on clear dental images  
- Not a replacement for professional dental diagnosis  

---

## 🔮 Future Improvements

- Use GANs or Diffusion Models  
- 3D tooth reconstruction  
- Better real medical dataset integration  
- Automated damage detection  

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 👨‍💻 Author

Taqi  
GitHub: https://github.com/Taqi12  

---

## 📜 License

This project is licensed under the MIT License.
