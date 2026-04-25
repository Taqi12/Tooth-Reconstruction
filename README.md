# 🦷 Dental Surface Reconstruction (AI-Based Inpainting System)

An advanced AI-powered dental reconstruction system that restores missing or damaged tooth surfaces from images. This project uses a hybrid approach combining classical computer vision, symmetry-based reconstruction, and optional Stable Diffusion inpainting.

---

## 🚀 Live Demo

Hugging Face Space: [https://taqijaved-tooth-reconstruction.hf.space]

---

## 📌 Overview

Dental reconstruction from images is challenging due to variations in angle, lighting, and tooth structure. Traditional auto-detection methods often fail when images are not perfectly aligned.

This project solves that problem by introducing a **user-guided mask painting approach**, where the user manually marks the missing tooth region for accurate reconstruction.

---

## 🧠 Key Innovation

Instead of automatic gap detection, this system uses:

- ✏️ User-painted mask (ImageEditor tool)
- 🔄 Symmetry-based tooth reconstruction
- 📋 Reference image patch transfer
- 🎨 LAB color matching
- 🔗 Poisson seamless blending
- 🔧 OpenCV refinement (Telea inpainting)
- 🤖 Optional Stable Diffusion inpainting (GPU)

---

## 🏗️ Pipeline

```
User Input (Painted Mask)
        ↓
Mask Extraction (Gradio ImageEditor)
        ↓
Reference Alignment (ORB + Homography)
        ↓
Symmetry / Reference Patch Selection
        ↓
Color Matching (LAB space)
        ↓
Seamless Blending (Poisson Clone)
        ↓
OpenCV Refinement (Inpainting)
        ↓
Optional Diffusion Enhancement
        ↓
Final Reconstructed Tooth Image
```

---

## 🖼️ Features

- 🎯 Manual gap selection using brush tool  
- 🧠 Smart reference selection (dual image input)  
- 🔄 Symmetry-based tooth reconstruction  
- 🎨 Advanced color correction (LAB matching)  
- 🔗 Seamless blending (Poisson blending)  
- 🔧 OpenCV inpainting fallback  
- 🤖 Stable Diffusion inpainting (optional GPU support)  
- ⚡ Real-time Gradio interface  

---

## ⚙️ Tech Stack

- Python  
- OpenCV  
- NumPy  
- PIL (Pillow)  
- Gradio  
- Diffusers (Stable Diffusion)  
- PyTorch  

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/dental-reconstruction.git
cd dental-reconstruction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```
gradio
numpy
opencv-python
pillow
torch
diffusers
transformers
accelerate
```

---

## ▶️ Run Locally

```bash
python app.py
```

Then open:

```
http://localhost:7860
```

---

## 🧪 How to Use

1. Upload **two healthy reference dental images**
2. Upload the **target image (with missing tooth)**
3. Use the **brush tool to paint the missing gap**
4. Click **Run Reconstruction**
5. View AI-generated reconstructed tooth

---

## 🧠 Model Behavior

The system dynamically selects the best reconstruction strategy:

- If GPU available → Stable Diffusion inpainting  
- Else → Symmetry-based reconstruction  
- Else → Reference patch copy  
- Else → OpenCV inpainting fallback  

---

## 📁 Project Structure

```
dental-reconstruction/
│
├── app.py                # Main Gradio application
├── model/                # (optional saved models)
├── utils/                # helper functions (if extended)
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- Not a medical diagnostic tool  
- Quality depends on reference images  
- Works best with clear dental images  
- Symmetry assumption may not always be perfect  

---

## 🔮 Future Improvements

- 3D tooth reconstruction (CT scan support)  
- GAN-based tooth synthesis  
- Automatic mask detection (optional mode)  
- Dental segmentation model integration  
- Clinical dataset training  

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.  
It is **not intended for clinical diagnosis or treatment planning**.

---

## 👨‍💻 Author

Taqi  
GitHub: https://github.com/Taqi12  

---

## 📜 License

MIT License
