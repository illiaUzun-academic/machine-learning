from transformers import pipeline

# Ініціалізація пайплайна для задачі відповіді на питання
qa_pipeline = pipeline("question-answering")

# Контекст та питання
context = """
The Great Wall of China is one of the greatest wonders of the world. It stretches over 21,000 kilometers (13,000 miles), 
crossing mountains, deserts, and grasslands. The Wall was originally built to protect Chinese states and empires against 
the raids and invasions of the various nomadic groups of the Eurasian Steppe.
"""
question = "What was the original purpose of the Great Wall of China?"

# Генерація відповіді
answer = qa_pipeline(question=question, context=context)

print(f"Питання: {question}")
print(f"Відповідь: {answer['answer']}")

# Цей код використовує модель з бібліотеки transformers для автоматичного визначення відповіді на задане питання,
# базуючись на наданому контексті. Це демонструє, як можна використовувати технології NLP для розуміння та взаємодії
# з текстовими даними на більш глибокому рівні.

