from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Create your views here.

def index(request):
    return render(request, 'pages/home.html')


def result(request):
    # Tải mô hình
    model_info = joblib.load('static/data/logistic_model_predic.pkl')
    model = model_info['model']

    # Lấy dữ liệu từ request GET
    val1 = float(request.GET.get('n1', 0))
    val2 = float(request.GET.get('n2', 0))
    val3 = float(request.GET.get('n3', 0))
    val4 = float(request.GET.get('n4', 0))
    val5 = float(request.GET.get('n5', 0))
    val6 = float(request.GET.get('n6', 0))
    val7 = float(request.GET.get('n7', 0))
    val8 = float(request.GET.get('n8', 0))

    # Thực hiện dự đoán
    prediction = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    context = {
        'result': 'Positive' if prediction == 1 else 'Negative',
    }
    return render(request, 'pages/result.html', context)
