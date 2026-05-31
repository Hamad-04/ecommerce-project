"""
admin.py - Django Admin Configuration
606315 - Programming for Data Science | Phase 3
"""

from django.contrib import admin
from .models import SalesRecord, PredictionLog


@admin.register(SalesRecord)
class SalesRecordAdmin(admin.ModelAdmin):
    """Admin panel for viewing Sales Records from the dataset."""
    list_display = ['invoice_no', 'description', 'quantity', 'unit_price', 'total_amount', 'country', 'invoice_date']
    list_filter = ['country']
    search_fields = ['invoice_no', 'description', 'stock_code']
    ordering = ['-invoice_date']
    readonly_fields = ['total_amount']


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    """Admin panel for viewing ML Prediction logs."""
    list_display = ['pk', 'quantity', 'unit_price', 'predicted_label', 'predicted_total', 'confidence', 'created_at']
    list_filter = ['predicted_label', 'created_at']
    ordering = ['-created_at']
    readonly_fields = ['created_at']
