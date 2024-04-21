from transformers import pipeline

# Ініціалізація пайплайна для аналізу настроїв з використанням моделі BERT
sentiment_pipeline = pipeline("sentiment-analysis")

# Текст для аналізу
texts = [
    "I love this product. It works great!",
    "This is a terrible product. It broke after one use.",
    "It's okay, but I've seen better."
]

# Аналіз настроїв для кожного тексту
for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"Текст: '{text}'\nНастрій: {result['label']}, Впевненість: {result['score']:.4f}\n")

# Цей код використовує модель BERT для визначення емоційної окраски тексту.
# В результаті ви отримаєте мітку настрою (наприклад, POSITIVE або NEGATIVE) і відсоток впевненості в цьому визначенні.
