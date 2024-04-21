from transformers import pipeline

# Ініціалізація пайплайна для сумаризації
summarizer = pipeline("summarization")

# Текст для сумаризації
text = """
The Amazon rainforest, also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometers (2,700,000 sq mi), of which 5,500,000 square kilometers (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and is estimated to have 390 billion individual trees divided into 16,000 species.
"""

# Генерація резюме
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

print("Сгенерированное резюме:")
print(summary[0]['summary_text'])

# Этот код использует предварительно обученную модель для генерации краткого содержания заданного текста.
# Модель стремится сохранить ключевые информационные точки текста, уменьшая его объем.
