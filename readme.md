# 🚚 Spotter Freight Rate Prediction - ML Assessment

**Author:** Shaden Abdul Majeed Al-Zin | AI & Data Science Engineer

## 📖 Project Overview
This repository contains an end-to-end Machine Learning pipeline built to predict freight load rates. Freight pricing is highly dynamic, depending on distance, truck equipment type, market conditions, and time. This project takes raw logistics data, cleans it, engineers intelligent features, and uses a robust machine learning model to predict pricing accurately.

## 🧠 The Challenges & The Engineering Approach
Real-world data is rarely perfect. Here is how the specific challenges in this dataset were handled so that even non-technical stakeholders can trust the pipeline:

1. **Highly Skewed Prices (The Outlier Problem):** 
   * **Issue:** Most freight rates are concentrated in a specific range, but a few extremely high prices skewed the data.
   * **Solution:** Applied a `log1p` transformation to normalize the prices during training, ensuring the model isn't confused by outliers, then reversed it (`expm1`) for the final predictions.
2. **Data Integrity Traps:** 
   * **Issue:** The dataset contained impossible negative weights (e.g., -47,500 lbs). 
   * **Solution:** Programmatically corrected these using absolute values during the preprocessing phase.
3. **High-Cardinality Locations:** 
   * **Issue:** There are thousands of unique pickup and delivery cities. Traditional One-Hot Encoding would flood the system with too many useless columns.
   * **Solution:** Implemented **Target Encoding**, allowing the model to smoothly learn the mathematical relationship between specific cities and their average historical prices.
4. **Incomplete Scenarios (The December Test):** 
   * **Issue:** The December evaluation data intentionally lacked key columns (like precise coordinates and market indices) that were present in the training data.
   * **Solution:** Built a dynamic fallback mechanism in the inference pipeline that securely injects the training medians for missing features, ensuring the model never breaks in production.

## 🏆 Model Selection
I evaluated two powerful ensemble models: **Random Forest** and **XGBoost**. 
* The script trains both models simultaneously and automatically selects the best performer based on the lowest Mean Absolute Error (MAE). 
* **XGBoost** won the evaluation with an MAE of ~$101, proving highly accurate for predicting highly variable freight costs.

## 🚀 How to Run the Project (Reproducibility)

### 1. Setup the Environment
Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.
```bash
# Install all required dependencies
pip install -r requirements.txt