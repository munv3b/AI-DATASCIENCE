# 🏠 California Housing Price Prediction — Regression Assignment

**Saylani AI & Data Science Program**

## 📌 Objective
Predict `median_house_value` (house price) for California districts using regression models, based on features like location, income, population, and housing characteristics.

## 📁 Files
| File | Description |
|---|---|
| `Regression_Assignment.ipynb` | Complete Jupyter notebook (EDA → Preprocessing → Modeling → Tuning) |
| `housing.csv` | Dataset used (California Housing Dataset) |
| `README.md` | This file |

## 🔧 Steps Performed
1. **Data Loading & Exploration** — shape, dtypes, statistical summary
2. **EDA** — missing value check, target distribution, correlation heatmap, outlier boxplots
3. **Preprocessing**
   - Missing values in `total_bedrooms` filled with **median**
   - Feature engineering: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
   - `ocean_proximity` encoded via **One-Hot Encoding**
4. **Feature Scaling** — `StandardScaler` applied (fit on train, transform on test)
5. **Train-Test Split** — 80/20 split, `random_state=42`
6. **Model Building** — 7 regression models trained and compared:
   - Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN
7. **Evaluation** — MAE, MSE, RMSE, R² Score calculated and compared for all models
8. **Hyperparameter Tuning** — `GridSearchCV` on Random Forest (best baseline model)

## 📊 Results

| Metric | Best Model (Tuned Random Forest) |
|---|---|
| MAE | ~32,259 |
| RMSE | ~50,124 |
| R² Score | **~0.808** |

`median_income` emerged as the strongest predictor of house value (confirmed by both correlation heatmap and feature importance analysis).

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Regression_Assignment.ipynb
```

## 🛠️ Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
