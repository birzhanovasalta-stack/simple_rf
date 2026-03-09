import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

st.title("📊 Прогноз оттока клиентов банка")
st.write("Введите данные клиента, чтобы узнать вероятность его ухода.")

# Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('churn_rf_model.joblib')

model = load_model()

# Создание интерфейса ввода
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Кредитный скоринг", min_value=300, max_value=850, value=600)
    age = st.slider("Возраст", 18, 100, 35)
    tenure = st.slider("Срок обслуживания (лет)", 0, 10, 5)
    balance = st.number_input("Баланс на счету", min_value=0.0, value=50000.0)

with col2:
    num_products = st.selectbox("Количество продуктов", [1, 2, 3, 4])
    has_card = st.selectbox("Есть кредитная карта?", ["Да", "Нет"])
    is_active = st.selectbox("Активный участник?", ["Да", "Нет"])
    complain = st.checkbox("Были жалобы?")

# Кнопка предсказания
if st.button("Рассчитать вероятность"):
    # Подготовка данных 
    features = np.array([[credit_score, age, tenure, balance, num_products, 
                          1 if has_card == "Да" else 0, 
                          1 if is_active == "Да" else 0,
                          1 if complain else 0]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Высокий риск ухода! Вероятность: {probability:.2%}")
    else:
        st.success(f"✅ Клиент лоялен. Вероятность ухода: {probability:.2%}")
