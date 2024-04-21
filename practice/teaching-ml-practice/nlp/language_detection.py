from langdetect import detect, DetectorFactory

# Задаємо однакове начальне значення для відтворюваності результатів
DetectorFactory.seed = 0

# Тексти для визначення мови
texts = [
    "Language detection is fascinating.",
    "La détection de la langue est fascinante.",
    "La detección de idiomas es fascinante.",
    "Языковое определение увлекательно.",
    "Мовне визначення захоплююче.",
    "Spracherkennung ist faszinierend."
]

# Визначення мови для кожного тексту
for text in texts:
    print(f"Текст: {text[:30]}... | Мова: {detect(text)}")

# Цей код демонструє, як можна використовувати бібліотеку langdetect для ідентифікації
# мови заданого тексту. Це може бути особливо корисно у многоязычных застосунках.
