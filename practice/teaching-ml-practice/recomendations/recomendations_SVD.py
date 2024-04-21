import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import SVD

# Датасет с оцінками пользователей
data = {
    'Користувач': ['Анна', 'Анна', 'Богдан', 'Богдан', 'Катерина', 'Катерина', 'Дмитро', 'Дмитро'],
    'Товар': ['Книга', 'Ноутбук', 'Книга', 'Телефон', 'Ноутбук', 'Телефон', 'Книга', 'Ноутбук'],
    'Оцінка': [5, 4, 4, 3, 5, 4, 5, 5]
}

# Перетворення датасету в формат, сумісний з бібліотекою Surprise
df = pd.DataFrame(data)
reader = Reader(rating_scale=(1, 5))  # Вказуємо діапазон оцінок
data = Dataset.load_from_df(df[['Користувач', 'Товар', 'Оцінка']], reader)

# Створення та тренування моделі SVD
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

# Тренуємо модель на всьому наборі даних
trainset = data.build_full_trainset()
svd.fit(trainset)

# Функція для генерації рекомендацій для конкретного користувача
def generate_recommendations(user, model, data, n=5):
    items = data['Товар'].unique()
    rated_items = data[data['Користувач'] == user]['Товар']
    items_to_predict = [item for item in items if item not in rated_items.values]

    predictions = [model.predict(user, item) for item in items_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions[:n]

# Генеруємо рекомендації для користувача "Анна"
recommendations = generate_recommendations('Анна', svd, df, n=3)

# Виводимо рекомендовані товари та передбачені оцінки
print([(pred.iid, pred.est) for pred in recommendations])
