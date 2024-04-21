from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer

# Текст для суммаризации
text = """
Artificial intelligence and machine learning are technologies that use vast amounts of data to learn and make decisions. 
AI is increasingly being used in industries like healthcare, where it can help predict patient outcomes and improve diagnostics. 
Machine learning, a subset of AI, enables computers to learn from data without being explicitly programmed. 
This technology has the potential to revolutionize many sectors by making processes more efficient and helping humans make better decisions.
"""

# Створення об'єкта для парсинга тексту
parser = PlaintextParser.from_string(text, Tokenizer("english"))

# Ініціалізація суммаризатора
summarizer = LuhnSummarizer()

# Визначення кількості речень у сумарному тексті
summary = summarizer(parser.document, 1)  # Кількість речень у сумарному тексті

# Виведення сумарного тексту
print("Сумарний текст:")
for sentence in summary:
    print(sentence)

# Цей код використовує LuhnSummarizer з бібліотеки sumy для автоматичного суммування тексту.
# Ви можете змінити кількість речень у сумарному тексті, відредагувавши аргумент у функції summarizer().
