from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Jalur file pkl Anda
MODEL_PATH = 'model/gradient_boosting_classifier_model.pkl'
SCALER_PATH = 'model/numerical_scaler.pkl'
ENCODER_PATH = 'model/categorical_encoder.pkl'

# Load model dan tools preprocessing
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, scaler, encoder]):
        return jsonify({'success': False, 'error': 'File pkl tidak lengkap di server!'}), 500
    
    try:
        data = request.get_json()
        
        # Ekstrak data waktu otomatis berdasarkan hari ini
        today = datetime.now()
        purchase_month = today.month
        purchase_day = today.day
        purchase_dayofweek = today.weekday()
        
        # 1. Bangun DataFrame dengan SEMUA kolom yang diinput dari Form secara dinamis
        input_df = pd.DataFrame([{
            # --- Fitur Numerik ---
            'price': float(data.get('price', 0)),
            'discount': float(data.get('discount', 0)),
            'final_price': float(data.get('final_price', 0)),
            'rating': float(data.get('rating', 0)),
            'review_count': int(data.get('review_count', 0)),
            'stock': int(data.get('stock', 0)),
            'seller_rating': float(data.get('seller_rating', 0)),
            'shipping_time_days': int(data.get('shipping_time_days', 0)),
            'purchase_month': purchase_month,
            'purchase_day': purchase_day,
            'purchase_dayofweek': purchase_dayofweek,
            
            # --- Fitur Kategorikal (Semuanya diambil langsung dari form input user) ---
            'category': data.get('category'),
            'subcategory': data.get('subcategory'),
            'brand': data.get('brand'),
            'location': data.get('location'),
            'device': data.get('device'),
            'payment_method': data.get('payment_method')
        }])
        
        # 2. Definisikan list nama kolom sesuai urutan eksak saat training (Cell 36)
        numerical_cols = [
            'price', 'discount', 'final_price', 'rating', 'review_count', 
            'stock', 'seller_rating', 'shipping_time_days', 
            'purchase_month', 'purchase_day', 'purchase_dayofweek'
        ]
        categorical_cols = ['category', 'subcategory', 'brand', 'location', 'device', 'payment_method']
        
        # 3. Jalankan transformasi scaling & encoding
        X_num = scaler.transform(input_df[numerical_cols])
        X_cat = encoder.transform(input_df[categorical_cols])
        
        # 4. Gabungkan secara horizontal (hstack) -> Menghasilkan struktur 56 fitur yang valid
        final_features = np.hstack((X_num, X_cat))
        
        # 5. Jalankan Prediksi
        prediction = model.predict(final_features)
        output = int(prediction[0])
        
        # Hitung Probabilitas Kepastian Klasifikasi AI
        try:
            probability = model.predict_proba(final_features)[0][output]
            probability = round(probability * 100, 2)
        except:
            probability = None

        result_text = "Dimungkinkan Dikembalikan (Returned)" if output == 1 else "Aman (Not Returned)"
        
        return jsonify({
            'success': True,
            'prediction': output,
            'result': result_text,
            'probability': probability
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)