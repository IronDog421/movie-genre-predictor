# Model Improvements for Better F1 Score 🚀

## Summary of Changes

This document outlines all the improvements made to boost the F1 score of the movie genre prediction model.

---

## 🔧 Key Improvements

### 1. **Enhanced TF-IDF Feature Engineering**

#### Word-Level TF-IDF:
- **Increased n-gram range**: `(1,2)` → `(1,3)` 
  - Now captures unigrams, bigrams, AND trigrams for better context
- **Lowered min_df**: `3` → `2`
  - Captures more rare but potentially informative terms
- **Increased max_features**: `300,000` → `500,000`
  - Richer feature representation
- **Added max_df**: `0.85`
  - Removes very common uninformative terms
- **Added strip_accents**: `'unicode'`
  - Better text normalization
- **Explicit lowercase**: `True`
  - Ensures consistent case normalization

#### Character-Level TF-IDF:
- **Adjusted n-gram range**: `(3,5)` → `(3,6)`
  - Captures longer character patterns
- **Lowered min_df**: `3` → `2`
- **Increased max_features**: `300,000` → `500,000`
- **Added max_df**: `0.85`
- **Added strip_accents**: `'unicode'`

### 2. **Optimized Logistic Regression Hyperparameters**

- **Increased C value**: `4.0` → `8.0`
  - Less regularization allows model to capture more complex patterns
- **Increased max_iter**: `2000` → `4000`
  - Better convergence, especially with more features
- **Added class_weight**: `'balanced'`
  - Handles class imbalance in multi-label classification
- **Added random_state**: `42`
  - Reproducibility

### 3. **Enhanced Threshold Calibration**

- **More granular search**: `19` → `50+` candidate thresholds
- **Extended quantile range**: `(0.05, 0.95)` → `(0.01, 0.99)`
- **Additional candidates**: 
  - Mean, median, 0.0, -0.5, 0.5
  - Covers more potential optimal thresholds per class
- **Unique candidates**: Removes duplicates for efficiency

---

## 🎯 Advanced Ensemble Approach (Optional)

For even better performance, an ensemble method is available that combines:

### Models in Ensemble:
1. **Logistic Regression** (weight: 0.5)
   - Primary model, typically best for text classification
   
2. **LinearSVC** (weight: 0.35)
   - Fast and effective for high-dimensional text data
   - Different decision boundary than LogReg
   
3. **Multinomial Naive Bayes** (weight: 0.15)
   - Good baseline, handles text probabilities well
   - Provides diversity in predictions

### Ensemble Benefits:
- **Combines strengths** of different algorithms
- **Reduces overfitting** through model averaging
- **More robust** predictions
- **Higher F1 scores** in most cases

---

## 📊 Expected Performance Improvements

### Original Model:
- micro-F1: ~0.633
- macro-F1: ~0.555

### Improved Single Model (Expected):
- micro-F1: **~0.65-0.68** (+2-5%)
- macro-F1: **~0.58-0.62** (+2-6%)

### Ensemble Model (Expected):
- micro-F1: **~0.67-0.70** (+4-7%)
- macro-F1: **~0.60-0.65** (+4-9%)

---

## 🚀 How to Use

### Run the Improved Single Model:
```python
# Execute cells in order up to and including the main training cell
# The model will automatically save the improved artifacts
```

### Run the Ensemble Model (Optional):
```python
# Execute the ensemble model cell
# This will train 3 models and combine their predictions
```

### Generate Predictions:
```python
# Single model
predict("../dataset_test.csv", "dataset_test_preds.csv", use_ensemble=False)

# Ensemble model
predict("../dataset_test.csv", "dataset_test_preds.csv", use_ensemble=True)
```

---

## 💡 Additional Tips for Further Improvement

1. **Data Augmentation**: Add more training data if possible
2. **Feature Engineering**: Add metadata features (year, director, actors)
3. **Deep Learning**: Consider BERT/RoBERTa for even better performance
4. **Hyperparameter Tuning**: Use GridSearchCV or Optuna for optimal parameters
5. **Class-specific Models**: Train specialized models for rare genres

---

## 📝 Notes

- Training time will increase due to more features and iterations
- The ensemble approach takes ~3x longer but provides better F1
- All models use `class_weight='balanced'` to handle imbalanced classes
- Threshold calibration is crucial for multi-label F1 optimization

---

## ✅ Validation

Compare your results using:
```python
from validator import compute_metrics
print(compute_metrics(y_va, pred))
```

This will show:
- Accuracy (exact match)
- F1 score (macro)
- Precision (macro)
- Recall (macro)
- Hamming loss
