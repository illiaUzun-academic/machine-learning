from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Цей приклад використовує попередньо навчену модель з бібліотеки transformers від Hugging Face, яка спеціалізується
# на роботі з моделями трансформерів. Модель, яка буде використовуватися, називається BLIP
# (Bootstrapped Language Image Pretraining), яка може генерувати описи для зображень.

# Ініціалізація моделі та процесора
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Функція для генерації опису зображення
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    # Підготовка вхідних даних
    inputs = processor(image, return_tensors="pt")

    # Генерація опису
    outputs = model.generate(**inputs)

    # Декодування та виведення результату
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    print("Generated caption:", caption)


# Приклад використання
generate_caption("test/side_eye_cat.jpg")

# У цьому коді ми спочатку ініціалізуємо процесор і модель для генерації описів зображень з використанням моделі BLIP
# від Hugging Face. Далі, функція generate_caption відкриває зображення, перетворює його у формат, який може бути
# оброблений процесором, та генерує опис зображення. Результатом виклику функції є згенерований текст, який описує зображення.
#
# Цей приклад демонструє базовий принцип генерації опису зображень і може бути використаний як вихідна точка для
# більш складних проєктів у цій області. Важливо зазначити, що для роботи з бібліотекою transformers та моделлю
# BLIP потрібно мати встановлений PyTorch або TensorFlow та саму бібліотеку transformers.
