import cv2
import matplotlib.pyplot as plt
import numpy as np


# Функція для виявлення синіх об'єктів на зображенні
def detect_blue_objects(image_path):
    # Завантаження зображення
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Встановлення діапазону синього кольору в HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Створення маски для виявлення синіх об'єктів
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Застосування маски до оригінального зображення
    blue_objects = cv2.bitwise_and(img, img, mask=mask)

    # Відображення оригінального зображення та зображення з виявленими синіми об'єктами
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(blue_objects, cv2.COLOR_BGR2RGB))
    plt.title('Blue Objects Detected')
    plt.axis('off')

    plt.show()


# Використання функції
detect_blue_objects('test/plate.jpg')

# У цьому прикладі ми спочатку перетворюємо зображення з BGR (стандартний формат кольору в OpenCV) до HSV
# (відтінок, насиченість, яскравість), оскільки HSV дозволяє нам більш інтуїтивно працювати з конкретними кольорами.
# Ми визначаємо нижню та верхню межі синього кольору в просторі HSV і створюємо маску, яка ідентифікує всі пікселі,
# що падають у цей діапазон кольору на зображенні. Потім ми застосовуємо цю маску до оригінального зображення,
# щоб отримати лише ті частини, які відповідають синім об'єктам. Нарешті, ми відображаємо оригінальне зображення
# та результат виявлення синіх об'єктів поряд для порівняння.
#
# Ця техніка може бути адаптована для виявлення об'єктів будь-якого кольору, просто змінивши діапазони HSV. Це робить
# її надзвичайно гнучкою для різноманітних застосувань, де потрібно ідентифікувати або відстежувати об'єкти певного кольору.