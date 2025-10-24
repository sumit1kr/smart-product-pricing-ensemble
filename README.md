# 🧠 Smart Product Pricing Challenge — Amazon ML Challenge 2025  
**Hackathon Platform:** Unstop  
**Team:** Zyro  
**Members:** Sumit Kumar (Leader), Ajitabh Ranjan, Harsh Raj, Disha Tribedy  

---

## 🏆 Problem Statement
**Goal:** Predict the product price using both **textual catalog data** and **image URLs**.  
**Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)  
**Dataset:**  
- `train.csv` → 75,000 samples  
- `test.csv` → 75,000 samples  
- Each record contains product name, description, image URL, and numerical attributes.

---

## 📂 Project Structure

```
Smart-Product-Pricing/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_feature_extraction.ipynb
│   ├── 3_model_training.ipynb
│   ├── 4_hyperparameter_tuning.ipynb
│   ├── 5_postprocessing.ipynb
│   └── 6_final_submission.ipynb
│
├── reports/
│   ├── main_report.tex
│   └── final_safe_submission.csv
│
└── requirements.txt
```

---

## ⚙️ Methodology Overview

### 🧩 1. Data Cleaning & Preprocessing
- Removed HTML tags, normalized case, and handled missing values.  
- Converted numeric columns to `float32` / `int16` for memory efficiency.  
- Applied `np.log1p()` to target (`price`).  
- Created an 90/10 train-validation split.

### 🔠 2. Feature Extraction
- **Text Features:** CLIP embeddings (384D → 32D PCA) + TF-IDF (15D).  
- **Image Features:** URL-based (domain, protocol, file type).  
- **Numeric Features:** `item_pack_qty`, `catalog_len`, and transformations.  
- Final selected features: **54** after reduction.

### ⚡ 3. Model Training
Trained an ensemble of models:
- **XGBoost:** 800 trees, tuned hyperparameters, early stopping.  
- **LightGBM:** fast gradient boosting variant.  
- **Ridge Regression:** L2 regularization (grid search).  
- **ElasticNet:** combined L1/L2 regularization (grid search).

### 🧮 4. Hyperparameter Tuning
- Manual and grid search for learning rate, max depth, regularization.  
- Early stopping for optimal generalization.  
- Tracked RMSE improvements per iteration.

### 🔗 5. Ensemble Strategy
Weighted blending using **inverse RMSE weighting**:

\[
w_i = \frac{1 / RMSE_i}{\sum_j 1 / RMSE_j}
\]

All predictions combined in **log-space** for better numerical stability.

| Model | RMSE | Weight |
|--------|------|--------|
| XGBoost | 36.75 | 0.35 |
| LightGBM | 36.93 | 0.35 |
| Ridge | 38.86 | 0.15 |
| ElasticNet | 38.86 | 0.15 |

---

### 🧹 6. Post-processing
- Clipped extreme prices (0.1–99.5th percentile).  
- Mean correction to align with training distribution.  
- Rounded to 2 decimals.  
- Verified all predictions are valid (no missing/negative values).

---

### 📊 7. Final Results
| Statistic | Training | Submission |
|------------|-----------|-------------|
| Mean Price | 23.65 | 23.65 |
| Median | 14.00 | 17.20 |
| Range | 2.48–72.05 | 2.48–72.05 |

✅ **Final submission:** `final_safe_submission.csv`  
✅ **Samples:** 75,000 complete predictions  
✅ **Validation:** All checks passed successfully  

---

## 🧠 Future Work
- Integrate **CLIP ViT-L** or **BLIP** for improved text-image representation.  
- Analyze feature impact using **SHAP**.  
- Cross-validation-based ensemble weighting.  
- End-to-end multimodal Transformer pipelines.  

---

## 💻 Technical Stack
- **Language:** Python 3.10  
- **Environment:** Google Colab (GPU enabled)  
- **Libraries:** NumPy, Pandas, Scikit-Learn, XGBoost, LightGBM, Transformers, Torch  

---

## 🚀 How to Run

### Step 1 — Clone Repository
```bash
git clone https://github.com/sumit1kr/Smart-Product-Pricing.git
cd Smart-Product-Pricing
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Execute Notebooks
```bash
# Data cleaning
jupyter nbconvert --execute notebooks/1_data_cleaning.ipynb

# Feature extraction
jupyter nbconvert --execute notebooks/2_feature_extraction.ipynb

# Model training
jupyter nbconvert --execute notebooks/3_model_training.ipynb

# Hyperparameter tuning
jupyter nbconvert --execute notebooks/4_hyperparameter_tuning.ipynb

# Post-processing & submission
jupyter nbconvert --execute notebooks/5_postprocessing.ipynb
```

### Step 4 — Output
Final predictions saved to:
```
reports/final_safe_submission.csv
```

---

## 📜 License
For educational and research use only.  
All data and competition content belong to **Amazon ML Challenge 2025** organizers.

---

