from spellchecker import SpellChecker

spell = SpellChecker()

# Приклад тексту з опечатками
text = "Speling erors in sentense are anoying."

# Розділення тексту на слова
words = text.split()

# Знаходження неправильних слів
misspelled = spell.unknown(words)

for word in misspelled:
    # Виправлення кожного неправильного слова
    correct_word = spell.correction(word)
    print(f"Неправильно: {word}, Виправлення: {correct_word}")

# Цей код використовує pyspellchecker для ідентифікації та виправлення опечаток у тексті.
# Він розділяє текст на слова, визначає неправильні слова та пропонує виправлення.
