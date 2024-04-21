import spacy

# Завантаження моделі для англійської мови
nlp = spacy.load("en_core_web_sm")

# Текст для обробки
text = "Apple announced the new iPhone 13 at their headquarters in Cupertino yesterday."

# Створення об'єкта Doc
doc = nlp(text)

# Виведення іменованих сущностей та їх типів
for ent in doc.ents:
    print(ent.text, ent.label_)

# Цей код використовує spaCy для визначення іменованих сущностей у тексті.
# Для кожної знайденої сущності виводиться її текст та тип (наприклад, ORG для організацій,
# PERSON для осіб, DATE для дат і т.д.).
