# 🏭 Industrial Predictive Maintenance

A Machine Learning project for predicting machine health, failures, failure types, Remaining Useful Life (RUL), and estimated repair costs.

## 📌 Project Overview

This project uses Machine Learning techniques to build multiple models that help predict and analyze potential machine failures.

The goal is to support **predictive maintenance** by identifying machine problems before they become serious, helping reduce unexpected downtime and maintenance costs.

## 🎯 Project Objectives

The project contains four main Machine Learning tasks:

### 1. Remaining Useful Life (RUL) Prediction

A regression model that predicts the estimated remaining operating time of a machine before maintenance or failure.

**Target:** `RUL`

### 2. Failure Within 24 Hours Prediction

A binary classification model that predicts whether a machine is likely to fail within the next 24 hours.

**Target:** `failure_within_24h`

* `0` → No failure expected within 24 hours
* `1` → Failure expected within 24 hours

### 3. Failure Type Prediction

A classification model that predicts the possible type of machine failure based on the current machine conditions.

**Target:** `failure_type`

### 4. Estimated Repair Cost Prediction

A regression model that estimates the expected repair cost based on machine and operational information.

**Target:** `repair_cost`

## 🤖 Machine Learning Models

Different Machine Learning algorithms were evaluated during the project, including:

* Linear Regression
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost

The best-performing model was selected for the corresponding prediction task.

## 📊 Dataset

The project uses the following dataset:

`predictive_maintenance_v3.csv`

The dataset contains industrial machine information and sensor/operational features used for predictive maintenance tasks.

## 📁 Project Structure

```text
Predictive-Maintenance/
│
├── models/
│   ├── rul_pipeline.pkl
│   ├── repair_cost_pipeline.pkl
│   ├── failure_type_pipeline.pkl
│   └── failure_24h_pipeline.pkl
│
├── notebooks/
│   ├── rul_hours.ipynb
│   ├── failure_within_24.ipynb
│   ├── failure_type.ipynb
│   └── estimated_repair_cost.ipynb
│
├── app.py
├── predictive_maintenance_v3.csv
├── Industrial_Predictive_Maintenance_Presentation.pptx
└── README.md
```

## 🖥️ Streamlit Application

The project includes a Streamlit application that provides an interactive interface for using the trained Machine Learning models.

To run the application:

```bash
streamlit run app.py
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit
* Joblib
* Jupyter Notebook

## 📈 Project Workflow

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Deployment with Streamlit
```

## 📂 Notebooks

The `notebooks` folder contains the development process for each prediction task, including data preprocessing, model training, evaluation, and analysis.

## 📑 Presentation

The project presentation is included in the repository:

`Industrial_Predictive_Maintenance_Presentation.pptx`

## 👨‍💻 Team

* **Ziad Mohamed**
* **Ziad Mahmoud**
* **Mohamed Sobhy**
* **Mohamed Gamil**

**Project:** Industrial Predictive Maintenance
**University:** Faculty of Computers and Artificial Intelligence – Benha University

---

⭐ If you find this project useful, feel free to explore the notebooks and models.
