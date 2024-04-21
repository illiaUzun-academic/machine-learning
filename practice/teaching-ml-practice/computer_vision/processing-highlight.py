import cv2
import numpy as np
import matplotlib.pyplot as plt


def highlight_color_in_image(image_path, color_range):
    # Завантаження зображення
    img = cv2.imread(image_path)

    # Конвертація зображення в HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Визначення діапазону кольору, який потрібно виділити
    # Наприклад, для виділення зеленого кольору: lower_green = np.array([40,40,40]), upper_green = np.array([80,255,255])
    lower_color = np.array(color_range[0])
    upper_color = np.array(color_range[1])

    # Створення маски для виділення певного кольору
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # Створення чорно-білого зображення
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_colored = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Застосування маски до чорно-білого зображення
    highlighted_img = np.where(mask[:, :, None] == 255, img, gray_img_colored)

    # Відображення результату
    plt.imshow(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB))
    plt.title('Highlighted Color Effect')
    plt.axis('off')
    plt.show()


# Використання функції
# Для прикладу, визначимо діапазон зеленого кольору
highlight_color_in_image('test/forest.jpg', ([40, 40, 40], [80, 255, 255]))

# У цьому прикладі ми спочатку конвертуємо зображення в HSV, оскільки це дозволяє нам легше працювати з
# кольоровими діапазонами. Далі створюємо маску, яка виділяє певний колір в зображенні за допомогою cv2.inRange().
# Після цього ми створюємо чорно-білу версію оригінального зображення і накладаємо на неї маску таким чином, що
# лише обрані кольорові об'єкти залишаються кольоровими, а решта зображення стає чорно-білою.
#
# Цей метод можна легко адаптувати для виділення різних кольорів, змінивши параметри кольорового діапазону в
# highlight_color_in_image функції. Такий підхід дозволяє створювати захоплюючі ефекти для фотографій, акцентуючи
# увагу на певних деталях зображення.