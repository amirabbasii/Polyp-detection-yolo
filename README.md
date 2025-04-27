# Polyp Detection using YOLO and Attention Mechanism

This project implements a deep learning-based solution for detecting polyps in colonoscopy images using the YOLO (You Only Look Once) model, enhanced with a lightweight attention mechanism. The attention mechanism improves the model's focus on critical regions in the images, leading to more accurate polyp detection.

## Project Overview

Colonoscopy is one of the primary methods for detecting polyps in the colon, which are potential indicators of colorectal cancer. Automated polyp detection can significantly assist in early diagnosis and improve the efficiency of medical professionals. This project aims to enhance polyp detection by leveraging YOLO for fast object detection, combined with an attention mechanism to focus on the most relevant features of the images.

### Key Features:
- **YOLO Model**: Utilizes the YOLOv8 architecture for real-time object detection.
- **Attention Mechanism**: Implements a lightweight attention model to refine the focus on relevant regions.
- **Multi-Dataset Approach**: The model is trained on multiple colonoscopy datasets, such as **Kvasir-SEG**, **CVC-ClinicDB**, **ETIS-LaribPolypDB**, and **CVC-ColonDB**.

## Datasets

This project uses the following datasets:
1. **Kvasir-SEG**
2. **CVC-ClinicDB**
3. **ETIS-LaribPolypDB**
4. **CVC-ColonDB**

The datasets contain images and ground truth masks, which are used to train and validate the model.

## Results
After training the model, the system can identify polyps in colonoscopy images, providing bounding boxes around the detected polyps. Below are the evaluation results on the test dataset:

| Metric       | Value |
|--------------|-------|
| **Precision (P)** | 0.963 |
| **Recall (R)**    | 0.917 |
| **mAP50**         | 0.981 |
| **mAP50-95**      | 0.787 |


