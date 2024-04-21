import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Завантаження датасету MNIST з рукописними цифрами
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Нормалізація зображень
train_images, test_images = train_images / 255.0, test_images / 255.0

# Ресайзинг зображень для CNN
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Створення моделі CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Тренування моделі
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Оцінка точності моделі
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Цей код створює та тренує модель CNN для класифікації рукописних цифр з датасету MNIST.
# Модель включає два згорткових шари для витягування особливостей з зображень, шар max
# pooling для зменшення розміру проміжних представлень зображень, повнозв'язний шар для
# класифікації, та вихідний шар з 10 нейронами, які відповідають 10 цифрам, з softmax активацією для отримання
# ймовірностей класів.
import numpy as np

# Виберіть індекс зображення, яке хочете класифікувати
img_index = 0  # Змініть це на будь-яке число від 0 до 9999

# Вибране зображення для класифікації
img = test_images[img_index]

# Робимо передбачення
predictions = model.predict(np.array([img]))

# Виводимо передбачення
plt.figure(figsize=(3, 3))
plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
plt.title(f'Predicted: {np.argmax(predictions)}, Actual: {test_labels[img_index]}')
plt.show()
