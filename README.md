# Multimodal Audio Risk Classification & Explanation

This project develops a **Transformer-based multimodal system** that classifies environmental sounds into risk levels (**Normal**, **Potential Threat**, **Danger**) and generates **natural language explanations**.

---

## Problem Motivation
Public safety monitoring often lacks real-time, human-readable explanations alongside detection.  
Our model not only **predicts risk levels** from audio but **explains** the context behind each prediction.

---

## Approach
- **Audio Preprocessing:**  
  - Raw audio converted to **log-mel spectrograms** using `Librosa`.
  - Metadata (location, label) used as additional features.

- **Two-Model Pipeline:**  
  - **Model 1:** Transformer-based risk classifier.  
  - **Model 2:** T5-based natural language explanation generator.

- **Training:**  
  - Hyperparameter tuning using **Optuna**.  
  - Risk classifier trained to 87.2% test accuracy.  
  - Explanation generator fine-tuned with **98% BERTScore F1**.

---

## Results
| Model              | Metric        | Value  |
|--------------------|---------------|--------|
| Risk Classifier    | Accuracy      | 87.2%  |
| Explanation Model  | BERTScore F1   | 98.0%  |

---

## Repository Structure
```
├── explanation_model_final/       # Explanation model code
├── processed_data/                 # Preprocessed spectrograms
├── FSD50K_Data.csv                 # Metadata CSV
├── audio_classification.ipynb      # Training risk classifier
├── best_multimodal_classifier.pth  # Saved best risk classifier
├── explanation_generation_data.csv # Data for explanation generation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
```

---

## Requirements
Install all libraries:
```bash
pip install -r requirements.txt
```

Main libraries used:
- `torch`
- `transformers`
- `librosa`
- `optuna`
- `scikit-learn`
- `evaluate`

---

## How to Run
1. Preprocess the audio into mel-spectrograms.
2. Train the **risk classifier** using `audio_classification.ipynb`.
3. Train the **explanation model** inside `explanation_model_final/`.
4. Evaluate performance on test datasets.

---

## Key Highlights
- Multimodal design combining audio signals and location metadata.
- Lightweight Transformer architecture for efficient deployment.
- Natural-language explanations boost model interpretability and human trust.

---


## Author

**ManiDatta**  
Data Science @ University of Colorado Boulder  
[GitHub](https://github.com/Manidatta1)
