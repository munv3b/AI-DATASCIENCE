# ❤️ Heart Failure Clinical Records — Classification Assignment

**Saylani AI & Data Science Program**

## 📌 Objective
Predict `DEATH_EVENT` (whether a patient survives the follow-up period) using clinical
records of 299 heart failure patients.

## 📁 Files
| File | Description |
|---|---|
| `Heart_Failure_Classification.ipynb` | Complete notebook (EDA → Preprocessing → Modeling → Tuning) |
| `heart_failure_clinical_records_dataset.csv` | Dataset used |
| `README.md` | This file |

## 🔧 Steps Performed
1. **Data Loading & Exploration** — shape, dtypes, statistical summary
2. **EDA** — missing value check, class distribution, correlation heatmap, boxplots by target
3. **Preprocessing** — no missing values, all columns already numeric (no encoding needed)
4. **Train-Test Split** — 80/20, stratified by target
5. **Feature Scaling** — `StandardScaler`
6. **Handling Imbalance** — **SMOTE** applied on training data only (203 vs 96, ~2:1 imbalance)
7. **Model Building** — 7 models compared: Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting, SVM, KNN, Naive Bayes
8. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Classification Report, Confusion Matrix
9. **Hyperparameter Tuning** — `GridSearchCV` on the actual best model (dynamically selected)

## 📊 Results

| Metric | Best Model (Tuned Random Forest) |
|---|---|
| Accuracy | 0.817 |
| Precision | 0.70 |
| Recall | 0.74 |
| F1-Score | 0.718 |

`time` (follow-up period) and `ejection_fraction` emerged as the strongest predictors of
`DEATH_EVENT` — consistent with clinical knowledge (low ejection fraction indicates weak
heart pumping capacity).

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook Heart_Failure_Classification.ipynb
```

## 🛠️ Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn (SMOTE)
