from openai import OpenAI
client = OpenAI(api_key='ВАШ_КЛЮЧ')

# # Ініціалізуйте початковий стан діалогу.
messages = [
  {"role": "system", "content": "You are a helpful assistant."}
]

while True:
  # Отримуємо введення від користувача.
  user_input = input("You: ")

  # Додаємо введення користувача в історію діалогу.
  messages.append({"role": "user", "content": user_input})

  # Робимо запит до API.
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
  )

  # Витягуємо відповідь асистента і виводимо її.
  assistant_response = response.choices[0].message.content
  print("Assistant:", assistant_response)

  # Додаємо відповідь асистента в історію діалогу.
  messages.append({"role": "assistant", "content": assistant_response})