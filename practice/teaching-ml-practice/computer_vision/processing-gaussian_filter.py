import cv2
import matplotlib.pyplot as plt


# Фільтр Гауса є популярним вибором для усунення шуму, оскільки він дозволяє досягти гладкості зображення, зберігаючи
# при цьому його загальні структури.

# Функція для усунення шуму з зображення
def remove_noise(image_path):
    # Завантаження зображення
    img = cv2.imread(image_path)

    # Перетворення зображення в відтінки сірого для спрощення
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Застосування Гаусового розмиття для усунення шуму
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Відображення оригінального та обробленого зображень
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title('Image after Noise Removal')
    plt.axis('off')

    plt.show()


# Використання функції
remove_noise('test/noisy.jpg')

# У цьому прикладі ми спочатку завантажуємо зображення та перетворюємо його в відтінки сірого, щоб спростити обробку.
# Далі застосовуємо Гаусове розмиття за допомогою функції cv2.GaussianBlur, де (5, 5) визначає розмір ядра розмиття, а
# 0 вказує на стандартне відхилення в горизонтальному та вертикальному напрямках. Ця операція допомагає усунути шум з
# зображення, роблячи його гладшим. На завершення, ми відображаємо оригінальне зображення та зображення після обробки для порівняння.
