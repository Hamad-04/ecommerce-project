"""
views.py - Django Views for E-Commerce Analytics
606315 - Programming for Data Science | Phase 3

Views:
  1. dashboard        - Data dashboard with charts and KPIs
  2. predict          - ML model prediction form
  3. prediction_history - Past prediction logs
  4. api_predict      - REST API endpoint (JSON)
  5. api_stats        - REST API for dashboard stats (JSON)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Avg, Sum
from django.conf import settings

from .models import SalesRecord, PredictionLog
from .forms import PredictionForm


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Load ML Model (cached at module level)
# ──────────────────────────────────────────────────────────────────────────────

_MODEL = None
_METRICS = None

def _load_model():
    """Load the serialised Scikit-Learn pipeline from disk (once)."""
    global _MODEL, _METRICS
    if _MODEL is None:
        model_path = os.path.join(
            settings.BASE_DIR, 'analytics', 'ml_model.pkl'
        )
        _MODEL = joblib.load(model_path)

    if _METRICS is None:
        metrics_path = os.path.join(
            settings.BASE_DIR, 'analytics', 'model_metrics.json'
        )
        with open(metrics_path) as f:
            _METRICS = json.load(f)

    return _MODEL, _METRICS


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Compute Dashboard Statistics from cleaned dataset
# ──────────────────────────────────────────────────────────────────────────────

def _get_dashboard_stats():
    """
    Read the cleaned dataset and compute KPIs and chart data.
    Returns a dictionary ready to pass as template context.
    """
    data_path = os.path.join(settings.BASE_DIR, 'analytics', 'cleaned_data.csv')
    df = pd.read_csv(data_path)

    # Filter valid records (positive quantity and price)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month

    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_revenue = df['TotalAmount'].sum()
    total_orders = df['InvoiceNo'].nunique()
    total_customers = df['CustomerID'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_products = df['StockCode'].nunique()

    # ── Monthly Sales ─────────────────────────────────────────────────────────
    monthly = df.groupby('Month')['TotalAmount'].sum()
    monthly_labels = [month_names[m] for m in monthly.index]
    monthly_values = [round(v / 1000, 2) for v in monthly.values]

    # ── Top 5 Countries ───────────────────────────────────────────────────────
    top_countries = df.groupby('Country')['TotalAmount'].sum().nlargest(5)
    country_labels = top_countries.index.tolist()
    country_values = [round(v / 1000, 2) for v in top_countries.values]

    # ── Top 10 Products ───────────────────────────────────────────────────────
    top_products = df.groupby('Description')['TotalAmount'].sum().nlargest(10)
    product_labels = [p[:35] + '...' if len(p) > 35 else p
                      for p in top_products.index]
    product_values = [round(v / 1000, 2) for v in top_products.values]

    # ── Revenue by Day of Week ────────────────────────────────────────────────
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_sales = df.groupby('DayOfWeek')['TotalAmount'].sum()
    dow_labels = [day_names[d] for d in dow_sales.index]
    dow_values = [round(v / 1000, 2) for v in dow_sales.values]

    return {
        # KPIs
        'total_revenue': f'{total_revenue:,.2f}',
        'total_orders': f'{total_orders:,}',
        'total_customers': f'{total_customers:,}',
        'avg_order_value': f'{avg_order_value:.2f}',
        'total_products': f'{total_products:,}',

        # Chart data (JSON strings for Chart.js)
        'monthly_labels': json.dumps(monthly_labels),
        'monthly_values': json.dumps(monthly_values),
        'country_labels': json.dumps(country_labels),
        'country_values': json.dumps(country_values),
        'product_labels': json.dumps(product_labels),
        'product_values': json.dumps(product_values),
        'dow_labels': json.dumps(dow_labels),
        'dow_values': json.dumps(dow_values),
    }


# ──────────────────────────────────────────────────────────────────────────────
# View 1: Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def dashboard(request):
    """
    Main data dashboard page.
    Shows KPI cards and interactive charts powered by Chart.js.
    """
    context = _get_dashboard_stats()
    context['page_title'] = 'Sales Dashboard'
    context['active_page'] = 'dashboard'
    return render(request, 'analytics/dashboard.html', context)


# ──────────────────────────────────────────────────────────────────────────────
# View 2: Prediction Page
# ──────────────────────────────────────────────────────────────────────────────

def predict(request):
    """
    Prediction page: users submit transaction features via a form,
    and the Scikit-Learn RandomForest model returns a prediction.
    """
    model, metrics = _load_model()
    prediction_result = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract cleaned form data
            quantity   = form.cleaned_data['quantity']
            unit_price = form.cleaned_data['unit_price']
            month      = form.cleaned_data['month']
            day_of_week = form.cleaned_data['day_of_week']
            hour       = form.cleaned_data['hour']

            # Build feature DataFrame and run inference (matches training feature names)
            feature_names = metrics.get('features', ['Quantity', 'UnitPrice', 'Month', 'DayOfWeek', 'Hour'])
            features = pd.DataFrame(
                [[quantity, unit_price, month, day_of_week, hour]],
                columns=feature_names
            )
            pred_label_num = model.predict(features)[0]
            pred_proba     = model.predict_proba(features)[0]

            label     = 'high' if pred_label_num == 1 else 'low'
            label_str = 'High Value Order 🚀' if label == 'high' else 'Low Value Order 📦'
            confidence = round(float(pred_proba.max()) * 100, 1)
            estimated_total = round(float(quantity * unit_price), 2)

            # Save prediction to database (ORM usage)
            log = PredictionLog.objects.create(
                quantity=quantity,
                unit_price=unit_price,
                month=month,
                day_of_week=day_of_week,
                hour=hour,
                predicted_label=label,
                predicted_total=estimated_total,
                confidence=confidence,
            )

            prediction_result = {
                'label': label,
                'label_str': label_str,
                'confidence': confidence,
                'estimated_total': estimated_total,
                'log_id': log.pk,
                'is_high': label == 'high',
            }
    else:
        form = PredictionForm()

    context = {
        'form': form,
        'prediction_result': prediction_result,
        'model_accuracy': metrics.get('accuracy', 0),
        'model_f1': metrics.get('f1_score', 0),
        'features': metrics.get('features', []),
        'page_title': 'Predict Order Value',
        'active_page': 'predict',
    }
    return render(request, 'analytics/predict.html', context)


# ──────────────────────────────────────────────────────────────────────────────
# View 3: Prediction History
# ──────────────────────────────────────────────────────────────────────────────

def prediction_history(request):
    """
    Display all past prediction requests stored in the PredictionLog model.
    Uses Django ORM to query the database.
    """
    logs = PredictionLog.objects.all()  # ORM query - returns all prediction logs

    # Aggregation stats using ORM
    total_predictions = logs.count()
    high_value_count  = logs.filter(predicted_label='high').count()
    low_value_count   = logs.filter(predicted_label='low').count()
    avg_confidence    = logs.aggregate(avg=Avg('confidence'))['avg'] or 0

    context = {
        'logs': logs,
        'total_predictions': total_predictions,
        'high_value_count': high_value_count,
        'low_value_count': low_value_count,
        'avg_confidence': round(avg_confidence, 1),
        'page_title': 'Prediction History',
        'active_page': 'history',
    }
    return render(request, 'analytics/prediction_history.html', context)


# ──────────────────────────────────────────────────────────────────────────────
# View 4: REST API - Predict (JSON)
# ──────────────────────────────────────────────────────────────────────────────

@csrf_exempt
def api_predict(request):
    """
    REST API endpoint for model prediction.
    Accepts POST with JSON body, returns JSON prediction.

    POST /api/predict/
    Body: {"quantity": 6, "unit_price": 2.55, "month": 12, "day_of_week": 2, "hour": 8}
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            quantity    = int(data.get('quantity', 1))
            unit_price  = float(data.get('unit_price', 1.0))
            month       = int(data.get('month', 1))
            day_of_week = int(data.get('day_of_week', 0))
            hour        = int(data.get('hour', 9))

            model, metrics = _load_model()
            feature_names = metrics.get('features', ['Quantity', 'UnitPrice', 'Month', 'DayOfWeek', 'Hour'])
            features = pd.DataFrame(
                [[quantity, unit_price, month, day_of_week, hour]],
                columns=feature_names
            )
            pred     = model.predict(features)[0]
            proba    = model.predict_proba(features)[0]

            label     = 'high' if pred == 1 else 'low'
            confidence = round(float(proba.max()) * 100, 1)

            return JsonResponse({
                'status': 'success',
                'prediction': label,
                'prediction_display': 'High Value Order' if label == 'high' else 'Low Value Order',
                'confidence': confidence,
                'estimated_total': round(quantity * unit_price, 2),
                'input': {
                    'quantity': quantity,
                    'unit_price': unit_price,
                    'month': month,
                    'day_of_week': day_of_week,
                    'hour': hour,
                },
            })
        except (KeyError, ValueError, TypeError) as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse(
        {'status': 'error', 'message': 'Only POST method is allowed.'},
        status=405
    )


# ──────────────────────────────────────────────────────────────────────────────
# View 5: REST API - Dashboard Stats (JSON)
# ──────────────────────────────────────────────────────────────────────────────

def api_stats(request):
    """
    REST API endpoint that returns dashboard statistics as JSON.
    GET /api/stats/
    """
    if request.method == 'GET':
        stats = _get_dashboard_stats()
        model, metrics = _load_model()
        return JsonResponse({
            'status': 'success',
            'kpis': {
                'total_revenue': stats['total_revenue'],
                'total_orders': stats['total_orders'],
                'total_customers': stats['total_customers'],
                'avg_order_value': stats['avg_order_value'],
                'total_products': stats['total_products'],
            },
            'model_performance': {
                'accuracy': metrics.get('accuracy'),
                'f1_score': metrics.get('f1_score'),
            },
            'total_predictions_logged': PredictionLog.objects.count(),
        })
    return JsonResponse({'status': 'error', 'message': 'GET only'}, status=405)
