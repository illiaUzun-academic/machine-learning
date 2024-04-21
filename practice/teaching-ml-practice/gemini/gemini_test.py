import google.generativeai as genai

GOOGLE_API_KEY = 'ВАШ_КЛЮЧ'
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model_name = 'gemini-1.5-pro-latest'

print(f"Using: {model_name}")
model = genai.GenerativeModel(model_name)

response = model.generate_content("What is the meaning of life?")
print(response.text)
