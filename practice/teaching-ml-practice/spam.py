# Імпортуємо необхідні бібліотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Завантажуємо датасет з текстовими повідомленнями
data = pd.read_csv('spam.csv', encoding='latin-1')

# Переглядаємо перші рядки датасету
print(data.head())

# Розділяємо датасет на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Векторизуємо текстові дані
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Створюємо модель Наївного Баєсового класифікатора
model = MultinomialNB()

# Навчаємо модель
model.fit(X_train_vec, y_train)

# Прогнозуємо мітки для тестового набору
predictions = model.predict(X_test_vec)

# Виводимо метрики для оцінки моделі
print("Точність моделі:", accuracy_score(y_test, predictions))
print("Звіт про класифікацію:\n", classification_report(y_test, predictions))