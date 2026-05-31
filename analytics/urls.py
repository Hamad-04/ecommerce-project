"""
urls.py - URL Routing for Analytics App
606315 - Programming for Data Science | Phase 3
"""

from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    # ── Main Pages ────────────────────────────────────────────────────────────
    path('', views.dashboard, name='dashboard'),                    # /
    path('predict/', views.predict, name='predict'),                # /predict/
    path('history/', views.prediction_history, name='history'),     # /history/

    # ── REST API Endpoints ────────────────────────────────────────────────────
    path('api/predict/', views.api_predict, name='api_predict'),    # /api/predict/
    path('api/stats/', views.api_stats, name='api_stats'),          # /api/stats/
]
