# 📊 Employee Attrition Prediction App

## 👋 Overview

This is a **Streamlit-based web application** that helps HR teams and analysts predict whether an employee is likely to leave the company. The app uses a pre-trained machine learning model and provides useful insights through visualizations and reports.

---

## 🎯 Objective

- Predict the likelihood of employee attrition using key input features.
- Support HR decision-making by identifying risk patterns.
- Visualize trends based on uploaded employee data.
- Export filtered attrition data reports.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – Web UI
- **Scikit-learn** – ML model training and prediction
- **Pickle** – Model and scaler loading
- **Pandas** – Data manipulation
- **Plotly** – Data visualization
- **NumPy** – Numeric computations

---

## 🔍 Features

- 📥 **Upload employee CSV data**  
  Load your dataset for analysis and filtering.
  ![image](https://github.com/user-attachments/assets/9c6adfd2-d887-424c-8bb4-8a7decd9d031)


- 📊 **Visual Analysis Dashboard**  
  Explore attrition rates by age, gender, and department.
  ![image](https://github.com/user-attachments/assets/9784e0db-9007-4d0b-8fff-a1d78b560afb)


- 🔮 **Attrition Prediction**  
  Input fields for Age, Income, Satisfaction, etc.  
  Uses a trained Decision Tree model to predict risk.
  ![image](https://github.com/user-attachments/assets/2881ab65-4971-4a1a-b831-fb42a8d548b3)


- 📤 **Download Report**  
  Filter and download employee data by attrition status.
  ![image](https://github.com/user-attachments/assets/7f4514a2-cc57-466c-ac9c-7d61325532b6)


---

## ⚙️ Model Inputs for Prediction

- Age  
- Monthly Income  
- Years at Company  
- Job Satisfaction (scale 1–4)  
- Department (One-hot encoded)  
- Marital Status (One-hot encoded)  
- Overtime (One-hot encoded)

The app scales numerical inputs using a pre-fitted scaler before passing them into the model.

---

## 🚀 How to Run

1. Make sure `emp.py`, model `.pkl` files, and scaler are available.
2. Install dependencies:
```bash
pip install streamlit pandas scikit-learn numpy plotly
```
3. Launch the app:
```bash
streamlit run emp.py
```

---

## 📂 Data Format for Upload

Your employee CSV should contain at least:
- `Age`, `Attrition`, `MonthlyIncome`, `Department`, `Gender`, `MaritalStatus`, `JobSatisfaction`, etc.

---

## 📜 License

Open-source project. Use and modify freely.

