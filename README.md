# E-Commerce Sales Analytics — Django Web App
## 606315 Programming for Data Science | Phase 3
**University of Petra | Student: Hamad Ali | ID: 202220558 | Instructor: Dr. Mohammad Arafah**

---

## Project Overview

A Django web application that:
1. **Dashboard** — visualizes E-Commerce KPIs and trends using Chart.js
2. **Predict** — serves a Scikit-Learn Random Forest model via a web form
3. **History** — logs all prediction requests in a SQLite database (Django ORM)
4. **REST API** — JSON endpoints at `/api/predict/` and `/api/stats/`

---

## Quick Start (Local)

```bash
# 1. Create & activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy dataset
cp cleaned_data_phase1.csv analytics/cleaned_data.csv

# 4. Run migrations
python manage.py makemigrations
python manage.py migrate

# 5. Create superuser (for admin panel)
python manage.py createsuperuser

# 6. Collect static files
python manage.py collectstatic --noinput

# 7. Run development server
python manage.py runserver

# Visit: http://127.0.0.1:8000/
# Admin: http://127.0.0.1:8000/admin/
```

---

## Deployment on PythonAnywhere (Free)

### Step 1 — Upload Files
Upload the entire project folder to PythonAnywhere via the Files tab.

### Step 2 — Create Virtual Environment
```bash
mkvirtualenv --python=/usr/bin/python3.10 ecommerce_env
pip install -r requirements.txt
```

### Step 3 — Configure Web App
- Go to **Web** tab → **Add a new web app**
- Choose **Manual configuration** → **Python 3.10**
- Set **Source code** to `/home/yourusername/ecommerce_project`
- Set **Working directory** to `/home/yourusername/ecommerce_project`

### Step 4 — Edit WSGI File
```python
import sys, os
path = '/home/yourusername/ecommerce_project'
if path not in sys.path:
    sys.path.insert(0, path)
os.environ['DJANGO_SETTINGS_MODULE'] = 'ecommerce_project.settings'
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

### Step 5 — Environment Variables
In **settings.py**, set:
```python
DEBUG = False
ALLOWED_HOSTS = ['yourusername.pythonanywhere.com']
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
```

### Step 6 — Static Files
In PythonAnywhere Web tab → Static files:
- URL: `/static/`
- Directory: `/home/yourusername/ecommerce_project/staticfiles`

Run: `python manage.py collectstatic`

### Step 7 — Migrate & Create Superuser
```bash
python manage.py migrate
python manage.py createsuperuser
```

### Step 8 — Reload App
Click **Reload** in the Web tab. Visit your live URL!

---

## Project Structure

```
ecommerce_project/
├── manage.py
├── requirements.txt
├── README.md
├── ecommerce_project/
│   ├── __init__.py
│   ├── settings.py          ← Project settings
│   ├── urls.py              ← Root URL configuration
│   └── wsgi.py              ← WSGI for deployment
├── analytics/               ← Main Django app
│   ├── __init__.py
│   ├── admin.py             ← Django Admin registration
│   ├── apps.py              ← App configuration
│   ├── forms.py             ← PredictionForm (Django Forms)
│   ├── models.py            ← SalesRecord + PredictionLog (ORM)
│   ├── views.py             ← 5 views: dashboard, predict, history, 2 APIs
│   ├── urls.py              ← App-level URL routing
│   ├── ml_model.pkl         ← Serialized Random Forest Pipeline (joblib)
│   ├── model_metrics.json   ← Accuracy, F1, confusion matrix
│   └── cleaned_data.csv     ← Cleaned dataset from Phase 1
├── templates/
│   ├── base.html            ← Base template with navbar & footer
│   └── analytics/
│       ├── dashboard.html   ← KPI + Chart.js charts
│       ├── predict.html     ← ML prediction form
│       └── prediction_history.html ← ORM query results table
└── static/
    └── analytics/
        ├── monthly_sales.png
        ├── top_countries.png
        ├── top_products.png
        └── confusion_matrix.png
```

---

## REST API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/stats/` | Dashboard KPIs as JSON |
| POST | `/api/predict/` | ML prediction endpoint |

### Example API Call
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"quantity": 6, "unit_price": 2.55, "month": 12, "day_of_week": 2, "hour": 8}'
```

---

## ML Model Details

| Item | Value |
|------|-------|
| Algorithm | Random Forest Classifier |
| Pipeline | StandardScaler + RandomForestClassifier(n_estimators=100) |
| Features | Quantity, UnitPrice, Month, DayOfWeek, Hour |
| Target | HighValue (1 if TotalAmount > £12.45 median) |
| Train/Test Split | 70% / 30% |
| Accuracy | 0.9999 |
| F1-Score | 0.9999 |
| Serialization | joblib (ml_model.pkl) |
