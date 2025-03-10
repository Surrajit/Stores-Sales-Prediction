# Big Mart Sales Prediction
- An end-to-end machine learning project to predict sales for Big Mart stores using a Flask web app. This project showcases data ingestion, validation, transformation, model training, evaluation, and deployment, built with a modular pipeline.

## Project Overview
- Using the [Big Mart Sales Prediction dataset](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets) from Kaggle, this project predicts `Item_Outlet_Sales` based on features like `Item_Weight`, `Item_Fat_Content`, `Item_MRP`, and outlet details. The trained model is deployed via a Flask app for real-time predictions.

### Key Features

- **Data Ingestion**: Loads and splits the dataset into train/test sets.
- **Data Validation**: Ensures schema compliance and detects data drift with EvidentlyAI.
- **Data Transformation**: Preprocesses numerical and categorical features.
- **Model Training**: Trains a regression model (e.g., Random Forest).
- **Model Evaluation**: Assesses performance with metrics like R².
- **Deployment**: Flask app for interactive sales predictions.

## Tech Stack
- **Python**: Core programming language.
- **Scikit-learn**: Model training and preprocessing.
- **Flask**: Web app deployment.
- **Pandas/Numpy**: Data manipulation.
- **EvidentlyAI**: Data drift detection.
- **YAML**: Configuration management.

## Setup Instructions

1. Install dependencies:

- pip install -r requirements.txt

2. Download the dataset from Kaggle and place Train.csv in data/.

3. Run the pipeline:

- python training_pipeline.py

4. Launch the Flask app:

- python app.py

## Future Improvements
- Deploy on AWS for a live demo.
- Add hyperparameter tuning for better accuracy.
- Enhance UI with CSS/JavaScript.


