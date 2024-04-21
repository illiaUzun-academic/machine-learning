import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Завантаження датасету Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Нормалізація зображень
train_images = train_images / 255.0
test_images = test_images / 255.0

# Ресайзинг зображень для CNN
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

# Створення моделі CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Тренування моделі
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Візуалізація результатів тренування
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

import numpy as np

# Імена класів у Fashion MNIST датасеті
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Вибір зображення з тестового набору
img_index = 5
img = test_images[img_index]

print("Shape of the image before expanding the dimensions:", img.shape)

# Додавання додаткового виміру до зображення, оскільки модель очікує на вхід пакет зображень
img = np.expand_dims(img, 0)

print("Shape of the image after expanding the dimensions:", img.shape)

# Передбачення класу зображення
predictions = model.predict(img)
score = tf.nn.softmax(predictions[0])

# Візуалізація результатів
plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.imshow(test_images[img_index].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Actual: {class_names[test_labels[img_index]]}")
plt.subplot(1,2,2)
plt.bar(range(10), score)
plt.title(f"Predicted: {class_names[np.argmax(score)]}")
plt.xticks(range(10), class_names, rotation=45)
plt.show()

print("\nThis image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score)))


# У цьому прикладі ми спочатку завантажуємо та підготовлюємо датасет Fashion MNIST, нормалізуючи зображення до діапазону
# [0, 1] і розширюючи розмірність зображень для відповідності вхідному формату CNN. Далі ми створюємо архітектуру CNN,
# яка складається з трьох згорткових шарів (Conv2D) з ReLU активацією та двох шарів максимального пулінгу (MaxPooling2D),
# за якими йде повнозв'язний шар (Dense). Модель компілюється з оптимізатором Adam та функцією втрат
# SparseCategoricalCrossentropy, що підходить для класифікації з множинними класами. Після тренування моделі на
# тренувальному наборі даних ми використовуємо її для оцінки точності на тестовому наборі.