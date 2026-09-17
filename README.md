# E-Commerce Order Cancellation & Return Analysis

A complete, end-to-end data science project that analyses **~50,000 real e-commerce orders** to identify cancellation and return patterns and predict order outcomes using machine learning.

---

## Project Structure

```
ecommerce_project/
├── backend/                        # (reserved for future microservice split)
├── data/
│   ├── clean_final_data.csv        # Merged dataset (49,222 rows)
│   ├── orders.csv                  # Raw orders (50,120 rows)
│   ├── customers.csv               # Customer profiles (10,000 rows)
│   ├── payments.csv                # Payment records (50,000 rows)
│   └── products.csv                # Product catalogue (20 products)
├── frontend/
│   └── index.html                  # Interactive single-page dashboard
├── model/
│   └── ecommerce_order_model.pkl   # Trained model bundle (generated)
├── report_images/                  # 13 EDA & model charts (generated)
├── model.py                        # Core pipeline: clean → EDA → train → save
├── train_model.py                  # Entry-point wrapper for model.py
├── app.py                          # Flask REST API backend
├── generate_report.py              # Word + HTML report generator
├── ecommerce_order_model.pkl       # Symlink / copy of trained model
├── E_Commerce_Order_Cancellation_Return_Report.docx  (generated)
├── ecommerce-project-report.html   (generated)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the model & generate EDA charts

```bash
cd ecommerce_project
python train_model.py
```

This will:
- Load and clean the data from `data/`
- Generate **13 visualisation images** in `report_images/`
- Train Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- Apply SMOTE to balance classes
- Save the best model bundle to `model/ecommerce_order_model.pkl`

### 3 — Generate reports

```bash
python generate_report.py
```

Produces:
- `E_Commerce_Order_Cancellation_Return_Report.docx` — professional Word report with all charts embedded
- `ecommerce-project-report.html` — self-contained HTML report with base64-embedded images

### 4 — Run the web application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

The dashboard provides:
- KPI summary cards (total orders, cancellation rate, return rate, revenue)
- Interactive charts: status distribution, payment methods, monthly trends, category breakdown
- **Predict** tab — enter order features and get a real-time cancellation/return probability

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | High-level dataset KPIs |
| GET | `/api/category_stats` | Orders by category and status |
| GET | `/api/monthly_trends` | Monthly order volumes by status |
| GET | `/api/payment_stats` | Payment method breakdown |
| POST | `/api/predict` | Single-order prediction |
| POST | `/api/batch_predict` | Batch prediction (JSON array) |

### Predict request example

```json
POST /api/predict
{
  "Age": 35, "Quantity": 2, "Discount": 10, "UnitPrice": 42,
  "Sales": 76, "Tenure": 180, "OrderMonth": 6, "OrderDOW": 2,
  "PaymentMethod": "Gateway", "Category": "Electronics",
  "CustomerSegment": "Regular", "City": "Tehran"
}
```

Response:
```json
{
  "prediction": 1,
  "label": "Cancelled/Returned",
  "probability_cancel_return": 67.4
}
```

---

## Machine Learning

**Target**: Binary — `1` = Cancelled or Returned, `0` = Completed

**Features**: Age, Quantity, Discount, UnitPrice, Sales, Tenure, OrderMonth, OrderDOW, PaymentMethod, Category, CustomerSegment, City

**Pipeline**:
1. Label-encode categorical features
2. SMOTE oversampling to balance classes
3. 80/20 train/test split (stratified)
4. StandardScaler normalisation
5. Train 4 models; compare ROC-AUC
6. Save best model bundle with `joblib`

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `clean_final_data.csv` | 49,222 | Pre-merged dataset with all features |
| `orders.csv` | 50,120 | Order transactions |
| `customers.csv` | 10,000 | Customer demographics |
| `products.csv` | 20 | Product catalogue |
| `payments.csv` | 50,000 | Payment records |

---

## Requirements

See `requirements.txt`. Main dependencies:

- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualisation
- **scikit-learn** — ML pipeline
- **xgboost** — gradient boosting
- **imbalanced-learn** — SMOTE resampling
- **flask**, **flask-cors** — REST API
- **joblib** — model serialisation
- **python-docx** — Word report generation

---

## Results Highlights

- Cancellation & return rates analysed across product categories, payment methods, customer segments, and cities
- XGBoost achieves the highest ROC-AUC among all tested models
- Key risk factors: product category, payment method, discount level, and customer segment
- Business recommendations provided in both the Word and HTML reports

---

*Generated with IBM Bob — E-Commerce Order Cancellation & Return Analysis Project*
