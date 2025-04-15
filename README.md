# Student Performance Machine Learning Application

This application provides a comprehensive machine learning toolset for analyzing student performance data through supervised and unsupervised learning techniques.

## Features

- **Data Exploration and Visualization**
  - Descriptive statistics
  - Distribution plots
  - Correlation analysis
  - Feature relationships

- **Supervised Learning**
  - Classification models (Decision Tree, Random Forest, SVM, Logistic Regression, KNN)
  - Regression models (Linear Regression, Decision Tree, Random Forest, SVR)
  - Hyperparameter tuning
  - Model evaluation metrics
  - Feature importance analysis

- **Unsupervised Learning**
  - Clustering (K-Means, Hierarchical)
  - Association rule mining
  - Visualization of clusters and rules

## Prerequisites

To run this application, you need:

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the source code.

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure you have the student performance dataset (`StudentsPerformance.csv`) in the correct location.

2. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

3. The application will open in your default web browser, typically at http://localhost:8501

## Project Structure

- `app.py`: Main entry point of the application
- `supervised.py`: Contains supervised learning implementations (classification and regression)
- `unsupervised.py`: Contains unsupervised learning implementations (clustering and association rules)
- `visualization.py`: Contains data exploration and visualization functions
- `utils.py`: Contains utility functions for data preprocessing and evaluation
- `StudentsPerformance.csv`: Dataset containing student performance data

## Dataset Description

The Student Performance dataset includes various attributes related to students' demographics, parental background, and test scores. The dataset helps in understanding factors that influence student academic performance.

Key features include:
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type (standard/free/reduced)
- Test preparation course completion
- Math, reading, and writing scores

## Usage Instructions

1. **Data Exploration**: Get insights into your dataset with visualizations and statistics.
2. **Supervised Learning**: 
   - For classification, predict categorical outcomes (e.g., pass/fail)
   - For regression, predict continuous values (e.g., test scores)
3. **Unsupervised Learning**:
   - Identify natural groupings in data with clustering
   - Discover relationships between features with association rules
   
## Customizing the Application

You can modify the application to work with different datasets:
1. Replace the dataset file
2. Update the data loading function in `utils.py`
3. Adjust the feature selection and preprocessing steps as needed

## Requirements

The application requires the following Python packages:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- scipy
- mlxtend
- statsmodels