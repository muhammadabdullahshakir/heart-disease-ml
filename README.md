# Heart Disease Detection using Machine Learning

## Overview
This project focuses on predicting the presence of heart disease in patients using machine learning algorithms. The dataset includes clinical features such as age, sex, blood pressure, cholesterol levels, and ECG results. The goal is to build and evaluate models that can help in early detection and prevention strategies.

## Dataset
- **Source:** Included in repository (`heart.csv`)
- **Rows:** 303 samples
- **Features:** 14 clinical attributes including `age`, `sex`, `cp` (chest pain type), `chol` (cholesterol), etc.
- **Target:** `target` (1 = heart disease present, 0 = not present)

## Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling
   - Train-test split

2. **Algorithms Used**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)

3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC curve

## Results

### Accuracy by Algorithm
| Algorithm           | Accuracy (%) |
|---------------------|--------------|
| Logistic Regression | 84           |
| SVM                 | 68           |
| KNN                 | 70           |
| Random Forest       | **88**       |
| Gradient Boosting   | 87           |
| Decision Tree       | 78           |

**Best Performer:** Random Forest (88% accuracy), followed closely by Gradient Boosting (87% accuracy).

### Performance Metrics
| Algorithm           | Precision | Recall   | F1 Score |
|---------------------|-----------|----------|----------|
| Logistic Regression | 0.898     | 0.822    | 0.859    |
| SVM                 | 0.758     | 0.673    | 0.713    |
| KNN                 | 0.771     | 0.692    | 0.729    |
| Random Forest       | 0.897     | 0.897    | 0.897    |
| Gradient Boosting   | 0.920     | 0.860    | 0.889    |
| Decision Tree       | 0.860     | 0.748    | 0.800    |

---

## Technologies Used
- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

**Future Work:**
- Hyperparameter tuning for underperforming models
- Incorporating additional patient features
- Exploring deep learning techniques
- Deploying the model for real-time healthcare applications


## How to Run
```bash
git clone https://github.com/your-username/heart-failure-prediction.git
cd heart-failure-prediction
jupyter notebook "Heart Disease Detection.ipynb"
jupyter notebook "Heart Disease Detection.ipynb"

