# Імпортуємо необхідні бібліотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантажуємо датасет Boston Housing
boston = load_boston()

# Перетворюємо дані у DataFrame для зручності роботи
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Виводимо перші 5 рядків датасету для огляду
print(data.head())

# Розділяємо датасет на вхідні характеристики (X) та цільову змінну (Y)
X = data.drop('PRICE', axis=1)
Y = data['PRICE']

# Розділяємо дані на навчальний та тестовий набори
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Нормалізуємо вхідні дані
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Створюємо модель лінійної регресії
lr = LinearRegression()

# Навчаємо модель на навчальному наборі даних
lr.fit(X_train_scaled, Y_train)

# Робимо прогнози на тестовому наборі
Y_pred = lr.predict(X_test_scaled)

# Виводимо метрики моделі
print("Середньоквадратична помилка (MSE):", mean_squared_error(Y_test, Y_pred))
print("Коефіцієнт детермінації (R^2):", r2_score(Y_test, Y_pred))

# Візуалізуємо реальні та прогнозовані значення
plt.scatter(Y_test, Y_pred)
plt.xlabel('Реальні ціни')
plt.ylabel('Прогнозовані ціни')
plt.title('Реальні vs Прогнозовані ціни')
plt.show()
