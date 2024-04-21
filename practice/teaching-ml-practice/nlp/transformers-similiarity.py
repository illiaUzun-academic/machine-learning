from sentence_transformers import SentenceTransformer, util

# Загрузка модели
model = SentenceTransformer('all-MiniLM-L6-v2')

# Тексты для сравнения
text1 = "Як я можу покращити свої навички гри на фортепіано?"
text2 = "Які стратегії допоможуть вам стати кращим у грі на фортепіано?"

# Преобразование текстов в векторы
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# Вычисление косинусного сходства
cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)

print(f"Косинусна подібність між текстами: {cosine_similarity.item():.4f}")

# Цей код використовує модель Sentence Transformers для перетворення текстів у вектори та визначення їх семантичного сходства.
# В результаті ми отримуємо міру косинусного сходства між векторами, яка вказує на ступінь семантичної близькості між текстами.
