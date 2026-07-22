# Model Performance Report

**Best model:** `gradient_boosting` — weighted F1 = 0.7571
**Build ID:** `local-codex-20260722`
**Git SHA:** `e279ef1`
**MLflow parent run:** `5276739b5d124a468859114f8a83031e`

---

## Decision Tree

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.7967   | - |
| Precision (w)   | 0.7089  | - |
| Recall (w)      | 0.7967     | - |
| F1 (weighted)   | 0.7465   | 0.7266 +/- 0.0348 |
| F1 Class 0      | 0.8863 | - |
| F1 Class 1      | 0.0395 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.83      0.95      0.89      1199
           1       0.09      0.03      0.04       237

    accuracy                           0.80      1436
   macro avg       0.46      0.49      0.46      1436
weighted avg       0.71      0.80      0.75      1436
```

---

## Random Forest

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.7716   | - |
| Precision (w)   | 0.7017  | - |
| Recall (w)      | 0.7716     | - |
| F1 (weighted)   | 0.7335   | 0.7293 +/- 0.0046 |
| F1 Class 0      | 0.8704 | - |
| F1 Class 1      | 0.0409 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.83      0.92      0.87      1199
           1       0.07      0.03      0.04       237

    accuracy                           0.77      1436
   macro avg       0.45      0.47      0.46      1436
weighted avg       0.70      0.77      0.73      1436
```

---

## Gradient Boosting

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.8294   | - |
| Precision (w)   | 0.6964  | - |
| Recall (w)      | 0.8294     | - |
| F1 (weighted)   | 0.7571   | 0.7586 +/- 0.0006 |
| F1 Class 0      | 0.9067 | - |
| F1 Class 1      | 0.0 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.83      0.99      0.91      1199
           1       0.00      0.00      0.00       237

    accuracy                           0.83      1436
   macro avg       0.42      0.50      0.45      1436
weighted avg       0.70      0.83      0.76      1436
```

---

## Xgboost

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.4687   | - |
| Precision (w)   | 0.6624  | - |
| Recall (w)      | 0.4687     | - |
| F1 (weighted)   | 0.5371   | 0.5574 +/- 0.0042 |
| F1 Class 0      | 0.6168 | - |
| F1 Class 1      | 0.1339 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.78      0.51      0.62      1199
           1       0.09      0.25      0.13       237

    accuracy                           0.47      1436
   macro avg       0.43      0.38      0.38      1436
weighted avg       0.66      0.47      0.54      1436
```

---

## Svm

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.6428   | - |
| Precision (w)   | 0.6694  | - |
| Recall (w)      | 0.6428     | - |
| F1 (weighted)   | 0.6558   | 0.6648 +/- 0.0029 |
| F1 Class 0      | 0.7816 | - |
| F1 Class 1      | 0.0191 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.80      0.77      0.78      1199
           1       0.02      0.02      0.02       237

    accuracy                           0.64      1436
   macro avg       0.41      0.39      0.40      1436
weighted avg       0.67      0.64      0.66      1436
```

---

## Neural Network

| Metric | Value | CV Mean +/- Std |
|---|---|---|
| Accuracy        | 0.64   | - |
| Precision (w)   | 0.6698  | - |
| Recall (w)      | 0.64     | - |
| F1 (weighted)   | 0.6545   | 0.6603 +/- 0.0075 |
| F1 Class 0      | 0.7793 | - |
| F1 Class 1      | 0.0227 | - |

**Classification Report:**
```
              precision    recall  f1-score   support

           0       0.80      0.76      0.78      1199
           1       0.02      0.03      0.02       237

    accuracy                           0.64      1436
   macro avg       0.41      0.39      0.40      1436
weighted avg       0.67      0.64      0.65      1436
```

---

