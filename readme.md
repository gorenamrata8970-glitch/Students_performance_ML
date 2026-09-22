### Model Summary

A Linear Regression model was developed to predict the number of student absences using demographic, academic, travel, activity, internet-access, and health-related features.

The dataset contained **395 student records** and **10 original variables**. After preprocessing, categorical variables were converted into numerical form using one-hot encoding. The dataset was divided into **80% training data and 20% testing data**.

The target variable was **`absences`**, while the predictor variables included:

* Age
* Sex
* School
* Parental status
* Travel time
* Study time
* Activities
* Internet access
* Health

The model was trained using **Linear Regression**.

### Model Evaluation

The model was evaluated using:

* **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted absences.
* **MSE (Mean Squared Error):** Measures the average squared prediction error.
* **RMSE (Root Mean Squared Error):** Represents the typical prediction error in the same unit as the target variable.
* **R² Score:** Measures how much of the variation in the target variable is explained by the model.

The actual metric values obtained from the model should be reported here:
MAE: 4.376057133194472
MSE: 36.3910959118619
RMSE: 6.032503287347791
R² Score: -0.015310469690154926

### Feature Interpretation

The fitted model produced both positive and negative coefficients. For example, `internet_yes` had a coefficient of approximately **3.12**, while `studytime` had a coefficient of approximately **-1.05**.

A positive coefficient indicates a positive association with predicted absences, while a negative coefficient indicates a negative association, while keeping the other variables constant.

These coefficients describe relationships found by the model and should not be interpreted as proof of causation.

### Conclusion

This project demonstrates how Linear Regression can be applied to student-related data to predict the number of absences. The preprocessing stage successfully converted categorical variables into numerical features, allowing them to be used by the regression algorithm.

The Actual vs Predicted plot and residual analysis were used to examine the model's predictions and errors. The model coefficients also provided an interpretation of how the selected features were associated with predicted absences.

Overall, the project demonstrates the complete supervised-learning workflow: **data preprocessing, feature selection, train-test splitting, model training, prediction, evaluation, visualization, and interpretation**.

**Note:** Since this dataset does not contain examination marks or a performance-score variable, the project is more accurately titled **"Student Absence Prediction Using Linear Regression"** rather than "Student Performance Prediction."
