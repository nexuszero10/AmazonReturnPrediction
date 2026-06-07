# Amazon Customer Return Prediction

This project explores the Amazon Customer Dataset Analysis, focusing on predicting whether a customer will return a product (`is_returned`) based on various product, customer behavioral, and marketplace features using Python. The analysis examines how pricing strategies, discounts, rating systems, and consumer logistics influence dynamic marketplace returns. Visualizations and machine learning models were implemented to uncover behavioral patterns within the data, providing an effective framework for business intelligence and recommendation pipelines.

Beyond the machine learning pipeline, this project includes a modern web application built with **Flask**, **Tailwind CSS**, and **Axios**, enabling users to perform real-time product return predictions through an interactive interface with a premium dark-mode dashboard.

---

## 🚀 Features

### Machine Learning Pipeline

- Handling high-cardinality categorical data
- Missing value analysis and data validation
- Outlier detection using IQR method
- Feature scaling using `StandardScaler`
- Categorical encoding using `OneHotEncoder`
- Multi-model classification benchmarking
- Model serialization using `Joblib`

---

### Interactive Web Application

- Modern responsive dark-mode UI (Tailwind CSS)
- Real-time prediction without page reload
- Asynchronous API requests using Axios
- Simple and clean user input form

---

### Data Visualization & Insight

- Feature distribution analysis
- Outlier detection using boxplots
- Target imbalance analysis
- Behavioral pattern exploration in customer returns

---

## 📊 Dataset & Feature Information

The dataset is a simulated Amazon-like e-commerce dataset containing structured transaction records.

| Feature Name | Type | Description |
|:------------|:-----|:------------|
| user_id | Categorical | Unique identifier for each customer |
| product_id | Categorical | Unique identifier for each product |
| category | Categorical | Product category |
| subcategory | Categorical | Sub-classification of product |
| brand | Categorical | Product brand |
| price | Numerical | Original product price |
| discount | Numerical | Discount percentage applied |
| final_price | Numerical | Price after discount |
| rating | Numerical | Product rating (1.7 - 5.0) |
| review_count | Numerical | Number of reviews |
| stock | Numerical | Inventory availability |
| seller_id | Categorical | Seller identifier |
| seller_rating | Numerical | Seller performance rating |
| purchase_date | Categorical | Transaction date |
| shipping_time_days | Numerical | Delivery time in days |
| location | Categorical | Customer location |
| device | Categorical | Device used for purchase |
| payment_method | Categorical | Payment method used |
| delivery_status | Categorical | Shipping outcome |
| is_returned | Target | Whether product was returned (True/False) |

---

## 🔍 Exploratory Data Analysis (EDA)

### 1. Missing Value & Data Quality Check

Data validation was performed to ensure consistency across numerical and categorical features.

**Insight:**
- Minimal missing values
- Strong variance in price-related and review-related variables

---

### 2. Outlier Detection (IQR Method)

Boxplot analysis was used to detect extreme values across numerical features.

**Insight:**
- `review_count` and `price` show high variance
- Some products receive unusually high engagement

---

### 3. Target Class Imbalance Analysis

The dataset exhibits imbalance between returned and non-returned products.

**Insight:**
- Majority of transactions are non-returned
- Requires balancing techniques for optimal classification

---

## 🤖 Machine Learning Models

### Logistic Regression
Baseline linear classifier used for comparison.

### Random Forest Classifier
Ensemble model capturing nonlinear relationships.

### Gradient Boosting Classifier
Final optimized model with best performance.

---

## 💾 Serialized Models

- `logistic_regression_model.pkl`
- `random_forest_classifier_model.pkl`
- `gradient_boosting_classifier_model.pkl`
- `numerical_scaler.pkl`
- `categorical_encoder.pkl`

---

## 🛠️ Tech Stack

### Backend & ML
- Python
- Flask
- Scikit-learn
- Joblib

### Frontend
- HTML
- Tailwind CSS
- JavaScript
- Axios

### Data Processing
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 💻 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/amazon-return-prediction.git
cd amazon-return-prediction
```

---

### 2. Create Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
```

---

### 4. Run Application

```bash
python app.py
```

---

### 5. Open in Browser

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
amazon-return-prediction/
│
├── images/
├── templates/
│   └── index.html
├── resources/
├── model/
├── app.py
├── amazon_ecommerce_1M.csv
├── logistic_regression_model.pkl
├── random_forest_classifier_model.pkl
├── gradient_boosting_classifier_model.pkl
├── numerical_scaler.pkl
├── categorical_encoder.pkl
└── README.md
```

---

## 🚀 Future Improvements

- SMOTE / class balancing optimization
- Hyperparameter tuning (GridSearchCV / RandomSearchCV)
- XGBoost / LightGBM integration
- SHAP explainability analysis
- Docker containerization
- Cloud deployment (AWS / Render / Railway)

---

## 👨‍💻 Author

**Tansah Jumeneng Prayogi**  
H1D023090  
Informatics Engineering  
Universitas Jenderal Soedirman

---

## 📜 License

This project is intended for educational and portfolio purposes.

© 2026 Tansah Jumeneng Prayogi. All Rights Reserved.