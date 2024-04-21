import spacy

# Завантаження моделі
nlp = spacy.load("en_core_web_sm")

# Аналізований текст
text = "Steve Jobs founded Apple Inc. in the United States."

# Обробка тексту
doc = nlp(text)

# Шукаємо співзасновників і компанії
for ent in doc.ents:
    # Якщо сущність - людина, шукаємо відношення "founded"
    if ent.label_ == "PERSON":
        for token in ent.root.head.children:
            if token.dep_ == "prep" and token.text == "in":
                for child in token.children:
                    if child.ent_type_ == "GPE":  # GeoPolitical Entity
                        print(f"{ent.text} founded a company in {child}")

# Цей код використовує spaCy для пошуку відношень типу "основав компанію в" між людьми і країнами.
# Він ітерує через сущності, знайдені в тексті, ідентифікує відношення за допомогою аналізу
# граматичних залежностей між словами.