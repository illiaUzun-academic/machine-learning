import pandas as pd

# Створення даних
data = {
    'Фільм': ['Фільм 1', 'Фільм 2', 'Фільм 3', 'Фільм 4', 'Фільм 5', 'Фільм 1', 'Фільм 2', 'Фільм 3'],
    'Оцінка': [5, 4, 4, 3, 5, 4, 5, 3]
}

df = pd.DataFrame(data)

# Розрахунок середньої оцінки для кожного фільму
average_ratings = df.groupby('Фільм')['Оцінка'].mean().sort_values(ascending=False)

# Виведення фільмів з найвищою середньою оцінкою
print(average_ratings)
