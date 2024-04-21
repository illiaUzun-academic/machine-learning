import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Приклад даних з фільмами та їх жанрами
data = {
    'Фільм': ['Фільм 1', 'Фільм 2', 'Фільм 3', 'Фільм 4', 'Фільм 5'],
    'Жанр': ['Комедія', 'Драма', 'Комедія', 'Драма', 'Драма'],
    'Оцінка': [5, 4, 4, 3, 5]
}

df = pd.DataFrame(data)

# Перетворення жанрів у векторну форму за допомогою One-Hot Encoding
df_onehot = pd.get_dummies(df, columns=['Жанр'])

# Видалення стовпця з назвами фільмів та оцінками для спрощення прикладу
df_onehot.drop(['Фільм', 'Оцінка'], axis=1, inplace=True)

# Виведення перетвореного DataFrame
print(df_onehot)

from sklearn.metrics.pairwise import cosine_similarity

# Обчислення косинусної схожості між фільмами
similarity = cosine_similarity(df_onehot)

# Перетворення матриці схожості в DataFrame для зручності
sim_df = pd.DataFrame(similarity, index=data['Фільм'], columns=data['Фільм'])
# Візуалізація матриці схожості
plt.figure(figsize=(10, 8))
sns.heatmap(sim_df, annot=True, cmap='coolwarm')
plt.title('Матриця косинусної схожості між фільмами')
plt.show()

# Виведення схожості фільмів до "Фільм 1"
print(sim_df['Фільм 1'])
