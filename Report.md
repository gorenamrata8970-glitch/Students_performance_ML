# Student Absence Prediction Using Linear Regression

## Project Overview

This project uses **Supervised Machine Learning** to predict the number of absences of a student using demographic, lifestyle, and academic-related attributes.

The project follows the complete machine learning workflow:

**Data Collection → Data Understanding → Data Preprocessing → Exploratory Data Analysis → Feature Encoding → Model Building → Model Evaluation → Interpretation → Conclusion**

The machine learning algorithm used in this project is **Linear Regression**.

---

## 1. Problem Statement

Student attendance can be influenced by several factors such as age, study time, travel time, health, internet access, and participation in activities.

The objective of this project is to develop a supervised machine learning model that predicts the **number of student absences** based on the available student-related attributes.

### Target Variable

**Absences**

The target variable represents the number of school absences recorded for each student.

### Machine Learning Type

**Supervised Learning – Regression**

Since the target variable is numerical, a regression algorithm is used.

---

## 2. Dataset Description

The dataset used in this project is `student_data.csv`.

It is based on the publicly available **UCI Student Performance Dataset**. The dataset used in this project is a selected subset containing student demographic, lifestyle, and academic-related attributes.

### Dataset Dimensions

* Number of observations: **395**
* Number of variables: **10**

### Features

| Feature    | Description                                 |
| ---------- | ------------------------------------------- |
| school     | Student's school                            |
| sex        | Student's gender                            |
| age        | Student's age                               |
| Pstatus    | Parent's cohabitation status                |
| traveltime | Travel time from home to school             |
| studytime  | Weekly study time                           |
| activities | Participation in extracurricular activities |
| internet   | Internet access at home                     |
| health     | Current health status                       |
| absences   | Number of school absences                   |

### Target Variable

`absences`

The model predicts the number of absences using the other available variables.

---

## 3. Data Preprocessing

### 3.1 Checking Data Types

The dataset contains both numerical and categorical variables.

#### Numerical Variables

* age
* traveltime
* studytime
* health
* absences

#### Categorical Variables

* school
* sex
* Pstatus
* activities
* internet

---

### 3.2 Missing Values

Missing values were checked using:

```python
print(df.isnull().sum())
```

The dataset was checked to ensure that missing values were identified before model building.

---

### 3.3 Duplicate Values

Duplicate records were checked using:

```python
print("Duplicate rows:", df.duplicated().sum())
```

Duplicate records were considered during data preprocessing.

---

### 3.4 Categorical Variable Encoding

Machine learning algorithms require numerical input. Therefore, categorical variables were converted into numerical form using **One-Hot Encoding**.

The following categorical features were encoded:

* school
* sex
* Pstatus
* activities
* internet

The encoding was performed using:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop="first", sparse_output=False)
X_cat_encoded = encoder.fit_transform(X_categorical)
```

The `drop="first"` option removes one category from each categorical variable to avoid redundant dummy variables.

After encoding, the categorical data contained **5 encoded features**.

---

### 3.5 Feature Matrix and Target

The numerical and encoded categorical variables were combined to create the final feature matrix.

```python
X = np.hstack([X_numerical, X_cat_encoded])
y = df["absences"]
```

Final dimensions:

* **X shape:** `(395, 9)`
* **y shape:** `(395,)`

---

## 4. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution of the data and relationships between variables.

---

### 4.1 Statistical Summary

The `describe()` function was used to obtain statistical measures.

Important observations include:

| Variable    |  Mean | Minimum | Maximum |
| ----------- | ----: | ------: | ------: |
| Age         | 16.70 |      15 |      22 |
| Travel Time |  1.45 |       1 |       4 |
| Study Time  |  2.04 |       1 |       4 |
| Health      |  3.55 |       1 |       5 |
| Absences    |  5.71 |       0 |      75 |

The average number of absences is approximately **5.71**.

---

### 4.2 Histogram of Absences

A histogram was used to understand the distribution of student absences.

```python
plt.figure(figsize=(7, 4))
plt.hist(df["absences"], bins=20)
plt.xlabel("Number of Absences")
plt.ylabel("Number of Students")
plt.title("Distribution of Student Absences")
plt.show()
```

The histogram helps identify the distribution and spread of absence values.

---

### 4.3 Boxplot of Absences

A boxplot was used to identify possible extreme values.

```python
plt.figure(figsize=(7, 4))
plt.boxplot(df["absences"])
plt.ylabel("Absences")
plt.title("Boxplot of Student Absences")
plt.show()
```

The boxplot helps identify potential outliers in the number of absences.

---

### 4.4 Study Time vs Absences

A scatter plot was used to examine the relationship between study time and absences.

```python
plt.figure(figsize=(7, 4))
plt.scatter(df["studytime"], df["absences"])
plt.xlabel("Study Time")
plt.ylabel("Absences")
plt.title("Study Time vs Absences")
plt.show()
```

This visualization helps determine whether a clear linear relationship exists between study time and absences.

---

### 4.5 Age vs Absences

A scatter plot was also created to observe the relationship between age and absences.

```python
plt.figure(figsize=(7, 4))
plt.scatter(df["age"], df["absences"])
plt.xlabel("Age")
plt.ylabel("Absences")
plt.title("Age vs Absences")
plt.show()
```

---

### 4.6 Correlation Analysis

Correlation was calculated for the numerical variables.

```python
correlation = df[
    ["age", "traveltime", "studytime", "health", "absences"]
].corr()

print(correlation)
```

A correlation heatmap was also used for visual representation.

```python
import seaborn as sns

plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

Correlation values help understand the strength and direction of linear relationships between numerical variables.

---

## 5. Model Building

### 5.1 Train-Test Split

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

The dataset was divided as follows:

* Training data: **80%**
* Testing data: **20%**

Approximately:

* Training samples: **316**
* Testing samples: **79**

---

### 5.2 Linear Regression

Linear Regression was selected because the target variable, `absences`, is numerical.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The trained model was then used to generate predictions.

```python
y_pred = model.predict(X_test)
```

---

## 6. Model Evaluation

The model was evaluated using four regression metrics:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Evaluation Results

| Metric   |      Result |
| -------- | ----------: |
| MAE      |  **4.3761** |
| MSE      | **36.3911** |
| RMSE     |  **6.0325** |
| R² Score | **-0.0153** |

### Metric Interpretation

#### MAE = 4.3761

The model's predictions differ from the actual number of absences by approximately **4.38 absences on average**.

#### MSE = 36.3911

The Mean Squared Error is **36.3911**. Since the errors are squared, larger prediction errors have a greater effect on this metric.

#### RMSE = 6.0325

The model has a typical prediction error of approximately **6.03 absences**.

#### R² Score = -0.0153

The R² score is negative and very close to zero. This indicates that the Linear Regression model does **not explain the variation in student absences well** on the test data.

The model performs slightly worse than a simple baseline that predicts the average number of absences for every student.

Therefore, the model should not be described as a highly accurate predictive model.

---

## 7. Actual vs Predicted Values

An Actual vs Predicted plot was created to visually compare the model predictions with the actual absence values.

```python
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Absences")
plt.ylabel("Predicted Absences")
plt.title("Actual vs Predicted Absences")

plt.show()
```

If the predictions were highly accurate, the points would be close to the diagonal reference line.

The obtained evaluation metrics indicate that the predictions have considerable variation from the actual values.

---

## 8. Residual Analysis

Residuals represent the difference between the actual and predicted values.

```python
residuals = y_test - y_pred
```

A residual plot was created to examine the prediction errors.

```python
plt.figure(figsize=(7, 5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0)

plt.xlabel("Predicted Absences")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.show()
```

Residual analysis helps identify patterns in the model's errors and determine whether a simple linear relationship is appropriate.

---

## 9. Feature Coefficient Interpretation

The Linear Regression coefficients were obtained as follows:

| Feature        | Coefficient |
| -------------- | ----------: |
| internet_yes   |      3.1183 |
| age            |      1.5661 |
| activities_yes |      0.5326 |
| traveltime     |      0.4311 |
| health         |     -0.0112 |
| studytime      |     -1.0502 |
| sex_M          |     -2.6867 |
| Pstatus_T      |     -4.4616 |
| school_MS      |     -4.6382 |

### Interpretation

A **positive coefficient** indicates that an increase or change in that feature is associated with an increase in the model's predicted absences, while keeping the other variables constant.

A **negative coefficient** indicates an association with lower predicted absences, keeping the other variables constant.

For example:

* `age` has a coefficient of approximately **1.57**, indicating a positive association with predicted absences.
* `studytime` has a coefficient of approximately **-1.05**, indicating a negative association.
* `internet_yes` has a coefficient of approximately **3.12** relative to students in the reference category (`internet_no`).
* `school_MS` has a coefficient of approximately **-4.64** relative to the reference category (`school_GP`).

### Important Note

These coefficients represent **associations within the fitted model**. They do not prove that a particular factor causes students to have more or fewer absences.

Because `drop="first"` was used during one-hot encoding, categorical coefficients are interpreted relative to their omitted reference categories.

---

## 10. Conclusion

This project developed a **Linear Regression model for predicting student absences** using demographic, lifestyle, and academic-related features.

The model achieved:

* **MAE:** 4.3761
* **MSE:** 36.3911
* **RMSE:** 6.0325
* **R²:** -0.0153

The results indicate that the selected variables and Linear Regression algorithm do not provide strong predictive performance for student absences.

The negative R² score suggests that the model performs slightly worse than a simple mean-based prediction on the test data.

This demonstrates an important aspect of machine learning: not every dataset and algorithm combination produces strong predictive performance. The results can be used to identify areas for improvement, such as selecting additional relevant features, applying different machine learning algorithms, or using a larger and more informative dataset.

---

## 11. Limitations

The project has several limitations:

1. The dataset contains only **10 selected variables**, which may not capture all factors influencing student absences.
2. The target variable, `absences`, can contain extreme values.
3. Linear Regression assumes a linear relationship between predictors and the target.
4. The model's negative R² indicates weak predictive performance.
5. The dataset does not contain student grade/performance variables, so this project focuses specifically on **absence prediction**, not academic performance prediction.
6. The results show association rather than causation.

---

## 12. Technologies Used

* Python
* Jupyter Notebook / JupyterLab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 13. Machine Learning Concepts Used

This project demonstrates the following concepts:

* Supervised Learning
* Regression
* Linear Regression
* Train-Test Split
* One-Hot Encoding
* Exploratory Data Analysis
* Correlation Analysis
* Data Visualization
* Model Prediction
* MAE
* MSE
* RMSE
* R² Score
* Residual Analysis
* Feature Coefficients
* Model Interpretation

---

## 14. Project Structure

```text
student-absence-prediction/
│
├── data/
│   └── student_data.csv
│
├── notebooks/
│   └── student_absence_prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 15. Dataset Source

The dataset is based on the **UCI Student Performance Dataset** from the UCI Machine Learning Repository.

UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/320/student+performance

The dataset used in this project is a selected 10-column subset of the publicly available student data.

---

## Final Project Summary

**Project Title:** Student Absence Prediction Using Linear Regression

**Problem:** Predict the number of student absences.

**Dataset:** Student data containing 395 observations and 10 variables.

**Target:** `absences`

**Algorithm:** Linear Regression

**MAE:** 4.3761

**MSE:** 36.3911

**RMSE:** 6.0325

**R²:** -0.0153

**Result:** The selected features and Linear Regression model show weak predictive performance for student absences.
