from transformers import pipeline, set_seed

# Ініціалізація пайплайна генерації тексту з використанням моделі GPT-2
text_generator = pipeline("text-generation", model="gpt2")

# Задання початкового фрагмента тексту
prompt = "In a distant future, humanity has discovered"

# Встановлення початкового значення для генератора випадкових чисел для відтворюваності результатів
set_seed(42)

# Генерація тексту на основі заданого фрагмента
generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)

print("Сгенерований текст:")
print(generated_text[0]["generated_text"])

# Цей код демонструє, як можна використовувати модель GPT-2 для генерації тексту на задану тему.
# Модель продовжує початковий фрагмент тексту, генеруючи відносно змістовний та послідовний текст.
