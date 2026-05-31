"""
forms.py - Django Forms for E-Commerce Analytics
606315 - Programming for Data Science | Phase 3
"""

from django import forms


MONTH_CHOICES = [
    (1, 'January'), (2, 'February'), (3, 'March'),
    (4, 'April'), (5, 'May'), (6, 'June'),
    (7, 'July'), (8, 'August'), (9, 'September'),
    (10, 'October'), (11, 'November'), (12, 'December'),
]

DAY_CHOICES = [
    (0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'),
    (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday'),
]


class PredictionForm(forms.Form):
    """
    Form for collecting transaction features for ML model prediction.
    Uses Scikit-Learn pipeline: StandardScaler + RandomForestClassifier.
    """

    quantity = forms.IntegerField(
        min_value=1,
        max_value=10000,
        label='Quantity',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 6',
        }),
        help_text='Number of units in the order (1–10,000).',
    )

    unit_price = forms.FloatField(
        min_value=0.01,
        max_value=50000,
        label='Unit Price (£)',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 2.55',
            'step': '0.01',
        }),
        help_text='Price per unit in British Pounds.',
    )

    month = forms.ChoiceField(
        choices=MONTH_CHOICES,
        label='Month',
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text='Month the transaction occurred.',
    )

    day_of_week = forms.ChoiceField(
        choices=DAY_CHOICES,
        label='Day of Week',
        widget=forms.Select(attrs={'class': 'form-select'}),
        help_text='Day the transaction occurred.',
    )

    hour = forms.IntegerField(
        min_value=0,
        max_value=23,
        label='Hour of Day',
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 9',
        }),
        help_text='Hour the transaction occurred (0–23, 24-hour format).',
    )

    def clean_month(self):
        return int(self.cleaned_data['month'])

    def clean_day_of_week(self):
        return int(self.cleaned_data['day_of_week'])
