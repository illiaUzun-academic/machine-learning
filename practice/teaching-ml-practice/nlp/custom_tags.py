from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

# Завантаження даних
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

X_train, X_test = newsgroups_train.data, newsgroups_test.data
y_train, y_test = newsgroups_train.target, newsgroups_test.target

# Створення моделі з використанням пайплайна
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(random_state=42)),
])

# Навчання моделі
model.fit(X_train, y_train)

# Тестування моделі
accuracy = model.score(X_test, y_test)
print(f"Точність моделі: {accuracy:.2f}")

# Використання моделі для автоматичного генерування тегів
text = "The CPU usage is high due to graphics rendering"
predicted_category = model.predict([text])[0]

print(f"Предиктована категорія: {newsgroups_train.target_names[predicted_category]}")

# Цей код демонструє базовий приклад того, як можна класифікувати текстові документи
# в предварительно определенные категории. Модель обучается на наборе данных 20newsgroups,
# а потім використовується для прогнозування категорії нового тексту.
