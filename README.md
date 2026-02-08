# Garbage Classification with Deep Learning

## 1. Project Overview

This project addresses the problem of **image-based garbage classification** using deep learning techniques.  
The goal is to automatically classify waste images into predefined categories, which is a relevant task for recycling automation and smart waste management systems.

The project is developed as part of a Deep Learning course and follows a structured, multi-stage workflow including data exploration, baseline modeling, and more advanced architectures.

---

## 2. Problem Definition

- **Task**: Supervised image classification  
- **Input (X)**: RGB images of garbage items  
- **Output (y)**: One of six garbage categories  

### Classes
- Cardboard  
- Glass  
- Metal  
- Paper  
- Plastic  
- Trash  

---
## 3. Dataset

The dataset used is the **Garbage Classification dataset** available on Kaggle:

https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

### Dataset characteristics
- ~2,500 images
- 6 classes
- JPEG format (`.jpg`)

The dataset is downloaded programmatically using the **Kaggle API** to ensure reproducibility.

---
## 4. Evaluation Metric

### Primary metric: **Accuracy**

Accuracy is defined as:


Accuracy = Number of correct predictions/Total number of predictions


Additionally, **confusion matrices** will be used for qualitative analysis.

## 5. Dataset Split

The dataset is split into three subsets using a stratified procedure to preserve class proportions:

- **Training set**: 70%  
- **Validation set**: 20%  
- **Test set**: 15%  

The split is performed using `train_test_split` from `scikit-learn` with a fixed random seed to ensure reproducibility.

---

## 6. State of the Art

Previous approaches for garbage image classification include both classical computer vision methods and deep learning techniques.

### Reported results on this dataset

https://mmcv.csie.ncku.edu.tw/~wtchu/papers/2020ICAN-meng.pdf

| Model | Approach | Pretrained | Metric | Reported Performance |
|------|--------|------------|--------|----------------------|
| SIFT + SVM | Classical CV | No | Accuracy | ~63% |
| Simple CNN | CNN from scratch | No | Accuracy | ~79% |
| HOG + CNN | Hybrid | No | Accuracy | ~81% |
| ResNet50 | Transfer Learning | Yes | Accuracy | ~91% |
| DenseNet121 | Transfer Learning | Yes | Accuracy | ~95% |
| MobileNetV2 | Transfer Learning | Yes | Accuracy | ~92–95% |
| SSL + Transformer | Self-supervised + DL | Yes | Accuracy | ~97% |
