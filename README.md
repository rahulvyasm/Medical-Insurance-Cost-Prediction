# Medical Insurance Cost Prediction

This repository contains Python code for predicting medical insurance costs using machine learning techniques. The project aims to analyze various factors influencing insurance charges, such as age, BMI, gender, smoking status, region, and charges and develop predictive models to estimate insurance costs accurately.

The dataset contains **2.7K rows** and **7 columns**
**Columns include**

1. Age
2. Sex
3. BMI (Body Mass Index)
4. Children
5. Smoker
6. Region
7. Charges

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Serialization](#model-serialization)
- [Contributors](#contributors)
- [License](#license)

## Introduction

Healthcare costs are a significant concern for individuals and families worldwide. Predicting medical insurance costs accurately can help insurance companies determine premiums and assist individuals in planning their healthcare expenses. This project focuses on building machine learning models to predict insurance costs based on demographic and health-related attributes.

## Problem Statement

1. What are the most important factors that affect medical expenses?
2. How well can machine learning models predict medical expenses?
3. How can machine learning models be used to improve the efficiency and profitability of health insurance companies?

## Features

- **Data Exploration**: Explore the dataset to understand its structure, identify missing values, and analyze the distribution of features.
- **Data Preprocessing**: Prepare the data by handling categorical variables, renaming columns, and scaling numerical features.
- **Model Training**: Utilize linear regression and ridge regression models to train predictive models on the prepared dataset.
- **Pipeline Construction**: Construct a data preprocessing pipeline to streamline the process of transforming input data for model training.
- **Model Evaluation**: Evaluate model performance using metrics such as R-squared score and mean squared error to assess predictive accuracy.
- **Model Serialization**: Save trained models and pipelines to disk using the pickle library for future use.

## Technologies Used

- **Python**: Programming language used for data manipulation, analysis, and model implementation.
- **Libraries**: NumPy, Pandas, Seaborn, Matplotlib, and Scikit-learn for data handling, visualization, and machine learning tasks.
- **Machine Learning Models**: Linear Regression, Ridge Regression
- **Pickle**: Python library used for serializing trained models and pipelines to disk.

## Usage

To use this project:

1. **Clone the Repository**: Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/rahulvyasm/Medical-Insurance-Cost-Prediction.git
   ```

2. **Install Dependencies**: Install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Data Preparation**: Prepare your data in CSV format similar to the provided `medical_insurance.csv` file.

4. **Run the Code**: Execute the Python scripts in your preferred environment to train models and perform predictions.

5. **Model Serialization**: Trained models and pipelines can be saved using pickle for future use.

## Installation

If you wish to run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/rahulvyasm/Medical-Insurance-Cost-Prediction.git
   ```

2. Navigate to the project directory:
   ```
   cd medical-insurance-prediction
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

Prepare your data in CSV format with columns representing features such as age, BMI, gender, smoking status, and region. Ensure that the data is clean and formatted correctly before proceeding with model training.

## Model Training

Train predictive models using linear regression and ridge regression techniques on the prepared dataset. Experiment with different hyperparameters and feature engineering methods to improve model performance.

## Model Evaluation

Evaluate model performance using appropriate metrics such as R-squared score and mean squared error. Compare the performance of different models and select the one that best suits your requirements.

## Model Serialization

Save trained models and preprocessing pipelines to disk using the pickle library. This allows you to reuse the models for future predictions without the need to retrain them every time.

## Contributors

- [M Rahul Vyas](https://github.com/rahulvyasm)

## License

This project is licensed under the [MIT License](https://github.com/rahulvyasm/Medical-Insurance-Cost-Prediction/blob/main/LICENSE). Feel free to use, modify, and distribute the code for your purposes.
