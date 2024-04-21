# Необхідні імпорти
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Створення зразка датасету з оцінками користувачів для фільмів
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [5, 3, 4, 2, 1],
    'Movie2': [4, 2, 5, 5, 2],
    'Movie3': [1, 4, 2, 3, 5],
    'Movie4': [5, 3, 3, 2, 4],
    'Movie5': [2, 5, 4, 4, 3]
}

# Конвертація даних у pandas DataFrame
df = pd.DataFrame(data).set_index('User')

# Нормалізація даних для забезпечення рівної важливості всіх ознак
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)

# Виконання кластеризації K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(df_normalized)

# Додавання міток кластерів до нашого оригінального DataFrame
df['Cluster'] = kmeans.labels_

# Розрахунок середніх оцінок для кожного фільму в кожному кластері
average_ratings = df.drop('Cluster', axis=1).groupby(df['Cluster']).mean()

# Вибірка топ-фільмів для рекомендації з кожного кластера
recommendations = average_ratings.apply(lambda x: x.nlargest(3).index.tolist(), axis=1)

print("Рекомендації для кожного кластера:")
print(recommendations)