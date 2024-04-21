import pandas as pd

# Створюємо приклад датасету з оцінками користувачів
data = {
    'Користувач': ['Анна', 'Анна', 'Богдан', 'Богдан', 'Катерина', 'Катерина', 'Дмитро', 'Дмитро'],
    'Товар': ['Книга', 'Ноутбук', 'Книга', 'Телефон', 'Ноутбук', 'Телефон', 'Книга', 'Ноутбук'],
    'Оцінка': [5, 4, 4, 3, 5, 4, 5, 5]
}

# Перетворюємо в DataFrame
df = pd.DataFrame(data)
print(df)

# Створюємо матрицю користувач-товар
user_item_matrix = df.pivot_table(index='Користувач', columns='Товар', values='Оцінка')

# Подивимося на створену матрицю
print(user_item_matrix)

from sklearn.metrics.pairwise import cosine_similarity

# Замінюємо NaN на 0
user_item_matrix_filled = user_item_matrix.fillna(0)

# Розраховуємо косинусну схожість
cosine_sim = cosine_similarity(user_item_matrix_filled)

# Перетворюємо отримані дані назад у DataFrame для зручності відображення
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix_filled.index, columns=user_item_matrix_filled.index)

# Подивимося на матрицю схожості
print(cosine_sim_df)

# Визначаємо користувача, для якого будемо робити рекомендації
target_user = 'Анна'

# Вибираємо користувачів, найбільш схожих на цільового, крім самого цільового користувача
similar_users = cosine_sim_df[target_user].sort_values(ascending=False).index[1:]

# Знаходимо товари, які оцінили схожі користувачі, але які не оцінив цільовий користувач
recommendations = set()
for user in similar_users:
    # Товари, які оцінив схожий користувач
    user_items = set(user_item_matrix.loc[user].dropna().index)
    # Виключаємо товари, які вже оцінив цільовий користувач
    target_user_items = set(user_item_matrix.loc[target_user].dropna().index)
    recommendations.update(user_items.difference(target_user_items))

# Показуємо рекомендації
print(recommendations)
