import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Датасет з книгами та їх жанрами
data = {
    'Книга': ['Книга 1', 'Книга 2', 'Книга 3', 'Книга 4', 'Книга 5'],
    'Жанри': ['Фантастика Пригоди', 'Фантастика Драма', 'Драма Детектив', 'Історія Драма', 'Пригоди Детектив']
}

df = pd.DataFrame(data)

# Створення TF-IDF вектора для жанрів (Term Frequency-Inverse Document Frequency)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Жанри'])

# Обчислення косинусної схожості
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Отримання рекомендацій для книги
def get_recommendations(title, cosine_sim=cosine_sim):
    # Отримання індексу книги, що відповідає заданому заголовку
    idx = df.index[df['Книга'] == title][0]

    # Отримання пар (індекс, схожість) для всіх книг з матриці схожості
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортування книг за ступенем схожості
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Отримання індексів найбільш схожих книг
    book_indices = [i[0] for i in sim_scores]

    # Повернення назв п'яти найбільш схожих книг
    return df['Книга'].iloc[book_indices]

# Приклад рекомендації
print(get_recommendations('Книга 1', cosine_sim))