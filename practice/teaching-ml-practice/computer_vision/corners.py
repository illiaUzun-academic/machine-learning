import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загрузка изображения
image = cv2.imread('test/image.png')

# Предобработка
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

# Нахождение контуров
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Аппроксимация контуров и преобразование в полигоны
polygons = [cv2.approxPolyDP(contour, epsilon=0.01*cv2.arcLength(contour, True), closed=True) for contour in contours]

# Визуализация
plt.figure(figsize=(300, 300))
for polygon in polygons:
    plt.plot(*zip(*np.append(polygon, [polygon[0]], axis=0)), marker='o')

plt.grid(True)
plt.title("Visualizing Polygons")
plt.savefig("test/corners.png")