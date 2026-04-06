# A Multi-Stage Computer Vision Approach for Food Image Analysis and Nutritional Estimation Using Depth-Guided Geometric Reasoning

## Overview

This repository implements a multi-stage computer vision approach for food image analysis and nutritional estimation from a single image. The method integrates object detection, classification, segmentation, and monocular depth estimation to enable end-to-end inference of food portion size and nutritional content. The approach combines semantic understanding with geometric reasoning to estimate volumetric properties of food items, which are further mapped to nutritional values using the USDA food database.

---

## Key Features

- Multi-stage pipeline for food analysis  
- Object detection using YOLOv8n  
- Fine-grained classification using EfficientNet-B3  
- Segmentation using Segment Anything Model (SAM)  
- Monocular depth estimation using MiDaS  
- Depth-guided portion estimation  
- Nutritional mapping using USDA database  
- Support for 40 food categories (Food-101 subset + Indian dishes)  

---

## Methodology

The pipeline processes an input image through the following stages:

1. **Food Detection** → YOLOv8n detects food items  
2. **Classification** → EfficientNet-B3 assigns labels  
3. **Segmentation** → SAM extracts pixel-level masks  
4. **Depth Estimation** → MiDaS predicts depth map  
5. **Fusion** → Mask + depth → volume estimation  
6. **Nutritional Mapping** → USDA database lookup  

---

## Repository Structure

    ├── services/
    │   ├── food_recognition.py
    │   ├── nutrition_calculator.py
    │
    ├── models/
    │   ├── train_yolo.ipynb
    │   ├── train_efficientnet.ipynb
    │   ├── food_yolov8.pt
    │   ├── classifier.pth
    │   ├── sam.pth
    │
    ├── dataset/
    │   └── food_dataset/
    │       ├── train/
    │       └── val/
    │
    ├── requirements.txt

---

## Installation

    git clone https://github.com/your-username/nutrivision.git
    cd nutrivision
    pip install -r requirements.txt

---

## Usage

### 1. Run Food Recognition

    from services.food_recognition import detect_and_classify

    results = detect_and_classify("image.jpg")

### 2. Compute Nutrition

    from services.nutrition_calculator import compute_nutrition

    nutrition = compute_nutrition(results)
    print(nutrition)

---

## Experimental Results

- Detection: mAP@0.5 = 0.821  
- Classification: Top-1 Accuracy = 90.22%  
- Portion Estimation: MRE = 11.7%  
- Nutritional Deviation: ~9–10%  
- Latency: 11.2 ms per image  

---

## Dataset

- 21 classes from Food-101  
- 19 custom Indian food classes  
- Total: 40 categories  

---

## Research Contribution

This work introduces a depth-guided geometric estimation approach by combining segmentation and monocular depth for portion estimation, enabling nutritional inference without requiring reference objects or additional hardware.

---

## Limitations

- Monocular depth provides relative estimation  
- Performance may vary under occlusion and complex backgrounds  
- Portion estimation depends on density assumptions  

---

## Future Work

- Domain-specific fine-tuning of SAM and MiDaS  
- Multi-view reconstruction for improved accuracy  
- Expanded dataset with ground-truth portion labels  

---

## License

This project is intended for academic and research purposes.