from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.exception import CustomException
from src.logger import logging

logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(),
    logging.FileHandler("flask_logs.log")
])

app = Flask(__name__)

MODEL = None
PREPROCESSOR = None

def load_models():
    global MODEL, PREPROCESSOR
    try:
        model_path = "api_assets/model.pkl"
        preprocessor_path = "api_assets/transformer.pkl"
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            logging.error(f"Preprocessor file not found: {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        logging.info(f"Loading model from {model_path}")
        MODEL = joblib.load(model_path)
        logging.info(f"Model loaded: {MODEL}")
        
        logging.info(f"Loading preprocessor from {preprocessor_path}")
        PREPROCESSOR = joblib.load(preprocessor_path)
        logging.info(f"Preprocessor loaded: {PREPROCESSOR}")
        
        return True
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return False

LOAD_SUCCESS = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():
    if not LOAD_SUCCESS:
        return render_template('error.html', error="Failed to load model or preprocessor. Check flask_logs.log.")
    
    if MODEL is None or PREPROCESSOR is None:
        return render_template('error.html', error="Model or preprocessor is None. Check logs.")
    
    if request.method == 'POST':
        try:
            item_weight = float(request.form.get('item_weight', 0)) if request.form.get('item_weight') else None
            item_fat_content = request.form['item_fat_content']
            item_visibility = float(request.form['item_visibility'])
            item_type = request.form['item_type']
            item_mrp = float(request.form['item_mrp'])
            outlet_year = int(request.form['outlet_year'])
            outlet_size = request.form.get('outlet_size', None)
            outlet_location = request.form['outlet_location']
            outlet_type = request.form['outlet_type']
            outlet_identifier = request.form['outlet_identifier']
            
            input_data = {
                "Item_Weight": item_weight,
                "Item_Fat_Content": item_fat_content,
                "Item_Visibility": item_visibility,
                "Item_Type": item_type,
                "Item_MRP": item_mrp,
                "Outlet_Establishment_Year": outlet_year,
                "Outlet_Size": outlet_size,
                "Outlet_Location_Type": outlet_location,
                "Outlet_Type": outlet_type,
                "Outlet_Identifier": outlet_identifier
            }
            df = pd.DataFrame([input_data])
            features = PREPROCESSOR.transform(df)
            prediction = MODEL.predict(features)[0]
            return render_template('index.html', prediction=f"Predicted Sales: {prediction:.2f}")
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return render_template('error.html', error=f"Prediction failed: {str(e)}")
    
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

