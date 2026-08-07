# ❤️ Framingham Heart Study — Classification Assignment

**Saylani AI & Data Science Program**

## 📌 Objective
Predict `TenYearCHD` (10-year risk of Coronary Heart Disease) using demographic, behavioral,
and medical data of 4238 patients from the Framingham Heart Study.

## 📁 Files
| File | Description |
|---|---|
| `Framingham_Classification.ipynb` | Complete notebook (EDA → Preprocessing → Modeling → Tuning) |
| `framingham.csv` | Dataset used |
| `README.md` | This file |

## 🔧 Steps Performed
1. **Data Loading & Exploration** — shape, dtypes, statistical summary
2. **EDA** — missing value check, class distribution, correlation heatmap, boxplots by target
3. **Preprocessing**
   - Missing values in `glucose`, `education`, `BPMeds`, `totChol`, `BMI`, `heartRate`,
     `cigsPerDay` filled with **median**
   - No extra encoding needed — all columns already numeric
4. **Train-Test Split** — 80/20, stratified by target
5. **Feature Scaling** — `StandardScaler`
6. **Handling Imbalance** — **SMOTE** applied on training data only (highly imbalanced,
   3594 vs 644, ~5.6:1)
7. **Model Building** — 7 models compared: Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting, SVM, KNN, Naive Bayes
8. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Classification Report, Confusion Matrix
9. **Hyperparameter Tuning** — `GridSearchCV` on the actual best model (dynamically selected —
   Logistic Regression turned out to be the best performer here, not Random Forest)

## 📊 Results

| Metric | Best Model (Tuned Logistic Regression) |
|---|---|
| Accuracy | ~0.66 |
| Precision | ~0.25 |
| Recall | ~0.60 |
| F1-Score | ~0.35 |

`age` and `sysBP` emerged as the strongest predictors of 10-year CHD risk.

**Note:** Overall scores are lower than typical because 10-year forward disease risk
prediction from limited clinical features is a genuinely hard problem — this is a
well-known characteristic of the Framingham dataset in published literature, not a
pipeline issue.

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook Framingham_Classification.ipynb
```

## 🛠️ Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imbalanced-learn (SMOTE)
