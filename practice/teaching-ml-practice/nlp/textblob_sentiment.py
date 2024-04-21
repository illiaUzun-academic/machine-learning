from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

import nltk
nltk.download('movie_reviews')

# Текст для аналізу
text = "I love this car. It's amazing!"

# Створення об'єкта TextBlob та використання аналізатора NaiveBayesAnalyzer
blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())

# Виведення результату аналізу тональності
print(blob.sentiment)

# Цей код аналізує тональність заданого тексту, використовуючи Naive Bayes аналізатор.
# Він повертає категорію (позитивний, негативний, нейтральний) та відсоткову впевненість в цій оцінці.
