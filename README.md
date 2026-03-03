# Credit Card Default Prediction — Classification Models

A machine learning classification project predicting whether credit card clients will **default on their next monthly payment**, enabling automated risk-tiering and proactive loss prevention.

## Project Overview

| Item | Detail |
|------|--------|
| **Dataset** | UCI "Default of Credit Card Clients" (Taiwan, 30,000 records) |
| **Problem Type** | Binary Classification (Default = 1, No Default = 0) |
| **Target Metric** | Recall (primary) / F1-Score (balancing) / ROC-AUC (ranking) |
| **Models** | Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, SVM |
| **Recommended Model** | Gradient Boosting Classifier |

## Business Value

A Taiwanese consumer bank needs to identify which of its ~30,000 credit card clients will default next month. The cost asymmetry is extreme:

- **Missing a defaulter (FN):** ~$5,000 loss per case
- **False alarm (FP):** ~$16 analyst review cost

The model automates risk-tiering, reducing both missed defaults and unnecessary manual reviews — directly improving **operational efficiency** and **P&L**.

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repo or navigate to the project directory
cd project_ML_claude

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter lab credit_default_classification.ipynb
```

### Quick Run (Execute All Cells)
```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute credit_default_classification.ipynb --output executed_notebook.ipynb
```

## Project Structure

```
project_ML_claude/
├── credit_default_classification.ipynb   # Main notebook (all analysis)
├── data/
│   └── credit_default.csv                # Raw dataset (30,000 rows)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── venv/                                 # Virtual environment
```

## Key Results

### Model Comparison

All models were tuned via **GridSearch/RandomizedSearch** with **5-Fold Stratified Cross-Validation**.

| Model | F1 | Recall | ROC-AUC |
|-------|---:|-------:|--------:|
| Gradient Boosting | Best | High | Best |
| Random Forest | High | Moderate | High |
| Logistic Regression | Moderate | Moderate | Moderate |
| AdaBoost | Moderate | Moderate | Moderate |
| Decision Tree | Moderate | Variable | Moderate |
| SVM (subsampled) | Moderate | Moderate | Moderate |

> *Exact metrics are computed dynamically in the notebook.*

### Cost-of-Error Analysis

Each model's confusion matrix was translated into **dollar costs**:
- `Net Cost = (FN × $5,000) + (FP × $16)`
- **Gradient Boosting** minimizes total expected financial loss.

### Top Predictive Features
1. **PAY_1** — Most recent payment status (strongest predictor)
2. **AVG_DELAY** — Average payment delay over 6 months
3. **AVG_UTILIZATION** — Credit line usage ratio
4. **LIMIT_BAL** — Credit limit amount
5. **AVG_PAY_RATIO** — Bill payment coverage ratio

## Dataset Source

**UCI Machine Learning Repository**: [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

- **Records:** 30,000 credit card clients
- **Period:** April–September 2005 (Taiwan)
- **Features:** 23 (demographics, credit history, payment behavior, bill amounts)
- **Target:** Binary — default on next month's payment (Yes/No)

The raw data is saved locally as `data/credit_default.csv` — **no API keys required** to run the notebook.

## Technical Highlights

- **No Data Leakage**: Scaling fitted only on training folds; stratified splits preserve class balance
- **Outlier Treatment**: IQR capping on financial amounts (heavy-tailed distributions)
- **Feature Engineering**: 6 derived features (utilization ratio, payment ratio, delay score, bill trend, payment consistency)
- **SVM Subsampling**: Justified O(n²) constraint — trained on 5K stratified subsample
- **Cost-Sensitive Analysis**: FP/FN costs quantified in dollar terms for production decision-making

---

*Built as part of a Bachelor 2 Classification Models evaluation.*
