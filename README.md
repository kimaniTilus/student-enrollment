# student-enrollment
Sure! Here's a simple, yet informative, **README.md** template for your GitHub project:

---

# Student Enrollment Prediction Model

## Project Overview
This project builds a machine learning model to predict which students are likely to enroll in a particular program and which ones may need additional support to graduate. It uses historical student enrollment data, academic records, and demographic information to make predictions. The model is based on a **Random Forest Classifier**.

## Features
- **Predict Enrollment**: Classifies whether a student is likely to enroll in a program or not.
- **Predict Graduation Success**: Helps identify students who might need additional support.
- **Model Evaluation**: The model is evaluated using accuracy, confusion matrix, ROC curve, and classification report.
- **Feature Importance**: Analyzes which features are most influential in predicting student enrollment.

## Dataset
- **Student Academic Records**: GPA, test scores, attendance rate.
- **Demographic Data**: Socioeconomic status, parental education level, participation in extracurriculars.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-enrollment-prediction.git
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

   **Requirements:**
   - pandas
   - scikit-learn
   - matplotlib
   - seaborn

## Usage

1. **Data Preparation**:
   - The dataset should be a CSV file with features like `high_school_gpa`, `test_score`, `attendance_rate`, etc.

2. **Train the Model**:
   Run the following script to train the Random Forest model:
   ```bash
   python train_model.py
   ```

3. **Evaluate the Model**:
   After training, the model is evaluated with metrics such as accuracy, confusion matrix, and ROC curve.

4. **Make Predictions**:
   You can use the trained model to predict whether a new student will enroll:
   ```python
   # Example of making predictions
   new_data = pd.DataFrame({
       'high_school_gpa': [3.5],
       'test_score': [1200],
       'attendance_rate': [95],
       'socioeconomic_status_Medium': [0],
       'socioeconomic_status_High': [1],
       'parent_education_College': [1],
       'parent_education_Graduate': [0],
       'extracurriculars_Yes': [1]
   })
   
   # Load the trained model
   import joblib
   model = joblib.load('student_enrollment_model.pkl')
   
   # Make a prediction
   prediction = model.predict(new_data)
   print(f"Prediction (1 = Enrolled, 0 = Not Enrolled): {prediction[0]}")
   ```

## File Structure
```
/student-enrollment-prediction
│
├── data/                  # Folder for raw and processed data
│   ├── students_data.csv  # Example dataset
│
├── train_model.py         # Script to train the model
├── test_model.py          # Script to test the model and evaluate performance
├── student_enrollment_model.pkl # Trained model (saved)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
```



