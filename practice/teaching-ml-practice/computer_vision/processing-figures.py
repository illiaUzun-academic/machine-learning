# Функція для виявлення кіл на зображенні
import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_circles(image_path):
    # Завантаження зображення
    img = cv2.imread(image_path, 0)  # Завантаження в градієнтах сірого для спрощення обробки
    img_original = cv2.imread(image_path)

    # Застосування розмиття для зменшення шуму
    img_blurred = cv2.GaussianBlur(img, (9, 9), 2)

    # Виявлення кіл за допомогою трансформації Хафа
    circles = cv2.HoughCircles(img_blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=40, minRadius=10, maxRadius=100)

    # Відображення знайдених кіл на оригінальному зображенні
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Намалювати зовнішнє коло
            cv2.circle(img_original, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Намалювати центр кола
            cv2.circle(img_original, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Відображення результату
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.axis('off')
    plt.show()


# Використання функції
detect_circles('test/money.jpg')