"""
models.py - Django ORM Models for E-Commerce Analytics
606315 - Programming for Data Science | Phase 3
"""

from django.db import models


class SalesRecord(models.Model):
    """
    Model representing a single sales transaction from the Online Retail dataset.
    Each instance corresponds to one row in the cleaned dataset.
    """
    invoice_no = models.CharField(max_length=20, verbose_name="Invoice Number")
    stock_code = models.CharField(max_length=20, verbose_name="Stock Code")
    description = models.CharField(max_length=300, verbose_name="Product Description")
    quantity = models.IntegerField(verbose_name="Quantity Sold")
    invoice_date = models.DateTimeField(verbose_name="Invoice Date")
    unit_price = models.FloatField(verbose_name="Unit Price (£)")
    customer_id = models.FloatField(null=True, blank=True, verbose_name="Customer ID")
    country = models.CharField(max_length=100, verbose_name="Country")
    total_amount = models.FloatField(verbose_name="Total Amount (£)", default=0.0)

    class Meta:
        verbose_name = "Sales Record"
        verbose_name_plural = "Sales Records"
        ordering = ['-invoice_date']

    def __str__(self):
        return f"{self.invoice_no} - {self.description[:40]} (£{self.total_amount:.2f})"

    def save(self, *args, **kwargs):
        """Auto-calculate TotalAmount before saving."""
        self.total_amount = self.quantity * self.unit_price
        super().save(*args, **kwargs)


class PredictionLog(models.Model):
    """
    Model to log prediction requests made through the web form.
    Stores the input features and the model's prediction result.
    """
    PREDICTION_CHOICES = [
        ('high', 'High Value Order'),
        ('low', 'Low Value Order'),
    ]

    # Input features
    quantity = models.IntegerField(verbose_name="Quantity")
    unit_price = models.FloatField(verbose_name="Unit Price (£)")
    month = models.IntegerField(verbose_name="Month (1-12)")
    day_of_week = models.IntegerField(verbose_name="Day of Week (0=Mon, 6=Sun)")
    hour = models.IntegerField(verbose_name="Hour of Day (0-23)")

    # Prediction result
    predicted_label = models.CharField(
        max_length=10,
        choices=PREDICTION_CHOICES,
        verbose_name="Prediction Result"
    )
    predicted_total = models.FloatField(verbose_name="Estimated Total (£)", default=0.0)
    confidence = models.FloatField(verbose_name="Model Confidence (%)", default=0.0)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Requested At")

    class Meta:
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"
        ordering = ['-created_at']

    def __str__(self):
        return (
            f"Prediction #{self.pk}: {self.get_predicted_label_display()} "
            f"(Qty={self.quantity}, Price=£{self.unit_price}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
