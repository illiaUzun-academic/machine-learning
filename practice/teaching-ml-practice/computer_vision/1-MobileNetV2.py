import numpy as np
from keras.preprocessing import image
from keras.src.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Класифікація зображень — це процес призначення категорії зображенню з певного набору категорій.
# Нижче наведено приклад, як можна використовувати попередньо навчену модель MobileNetV2 для класифікації зображень.
# https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c

# Завантаження попередньо навченої моделі MobileNetV2
model = MobileNetV2(weights='imagenet')


# Функція для класифікації зображення
def classify_image(img_path):
    # Завантаження зображення, його перетворення до необхідного розміру та препроцесинг
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    preprocessed_image = preprocess_input(img_array_expanded_dims)

    # Передбачення моделі
    predictions = model.predict(preprocessed_image)

    # Декодування передбачень
    return decode_predictions(predictions, top=3)[0]


# Приклад використання
result = classify_image('test/side_eye_cat.jpg')
for i, (imagenet_id, label, score) in enumerate(result):
    print(f"{i + 1}: {label} ({score * 100:.2f}%)")

# У цьому коді ми використовуємо попередньо навчену модель MobileNetV2 з вагами imagenet для класифікації зображень.
# Модель MobileNetV2 ефективна та швидка, що робить її підходящою для використання в мобільних додатках або на пристроях
# з обмеженими обчислювальними ресурсами. Ми здійснюємо препроцесинг зображення перед його подачею в модель і декодуємо
# передбачення, щоб отримати назви класів та їх вірогідності.
