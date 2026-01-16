# 🔋 NASA Li-Ion Battery RUL Prediction with XAI & Adaptive LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview
This project focuses on predicting the **Remaining Useful Life (RUL)** of Li-Ion batteries using the **NASA PCoE Dataset**. Unlike traditional approaches, this project introduces two key innovations:
1.  **Explainable AI (XAI):** An Attention Mechanism that visualizes which time steps the model focuses on.
2.  **Adaptive Input Strategy:** A novel padding technique to handle "Cold Start" problems for short-life batteries (e.g., Battery B0032).

## 📊 Key Features
* **Hybrid Architecture:** LSTM + Attention Mechanism.
* **Adaptive Windowing:** Capable of predicting RUL from Cycle 0 (Solving the cold-start problem).
* **Robustness:** Tested under different thermal conditions (4°C, 24°C, 43°C).
* **High Accuracy:** Reduced RMSE for critical cases (B0032) by **~84%**.

## 📂 Project Structure
```bash
├── data/               # Raw and processed datasets (Ignored in repo)
├── figures/            # Generated plots (Loss curves, Predictions, Heatmaps)
├── models/             # Saved PyTorch models (.pth)
├── notebooks/          # Jupyter Notebooks for EDA and Training
│   ├── 01_Data_Exploration_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Evaluation_and_XAI.ipynb
├── src/                # Source Code
│   ├── data_loader.py  # Loads NASA .mat files
│   ├── feature_eng.py  # Feature extraction pipeline
│   ├── model.py        # Attention-LSTM Architecture
│   ├── train.py        # Training loop
│   └── evaluate.py     # Testing & Visualization
└── README.md           # This file
```
# 🚀 Installation & Usage
## Clone the Repository
```Bash
git clone [https://github.com/YOUR_USERNAME/NASA_LiIon_XAI_RUL.git](https://github.com/YOUR_USERNAME/NASA_LiIon_XAI_RUL.git)
cd NASA_LiIon_XAI_RUL
```
## Install Dependencies
```Bash
pip install torch pandas numpy matplotlib seaborn scikit-learn tqdm scipy
```
## Run Training
```Bash
python src/train.py
```
## Evaluate & Visualize
```Bash
python src/evaluate.py
```
# 📈 Results
### 1. Training Performance (Loss Curve)
The model shows stable convergence with mini-batch training.

### 2. Adaptive Prediction (Battery B0032)
Comparison of Fixed Window vs. Adaptive Input Strategy.

### 3. Explainability (Attention Map)
The model focuses on recent cycles to determine the degradation trend.
