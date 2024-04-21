import cv2
import matplotlib.pyplot as plt


# Функція для перетворення зображення в градієнти сірого і застосування порогової обробки
def apply_threshold(image_path):
    # Завантаження зображення
    img = cv2.imread(image_path)

    # Перетворення зображення в градієнти сірого
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Застосування порогової обробки
    _, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

    # Відображення оригінального зображення, зображення в градієнтах сірого, та після порогової обробки
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(threshold_img, cmap='gray')
    plt.title('Thresholded Image')
    plt.axis('off')

    plt.show()


# Використання функції
apply_threshold('test/sniper_king.png')

# У цьому прикладі ми спочатку завантажуємо зображення та перетворюємо його в градієнти сірого за допомогою функції
# cv2.cvtColor з параметром cv2.COLOR_BGR2GRAY. Після цього застосовуємо порогову обробку з використанням функції
# cv2.threshold, де 120 — це порогове значення інтенсивності, а 255 — значення, яке присвоюється пікселям,
# що перевищують поріг. Режим cv2.THRESH_BINARY означає, що пікселі з інтенсивністю вище порога стануть білими,
# а всі інші — чорними, створюючи бінарне зображення.
#
# Такий підхід може бути корисний для виділення певних елементів зображення, спрощення зображення для подальшого
# аналізу або підготовки даних для машинного навчання. Порогова обробка є фундаментальною технікою в обробці зображень
# і має широкий спектр застосувань.