import cv2

# Завантаження моделі виявлення облич
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Виявлення об'єктів полягає в ідентифікації та локалізації одного або кількох об'єктів на зображенні.
# Ось як можна використовувати OpenCV для виявлення облич на зображенні:

# Функція для виявлення облич
def detect_faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Виявлення облич
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Намалювання рамок навколо облич
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Встановлення вікна на певну позицію
    window_name = 'Faces'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 100, 100)  # Змініть значення 100, 100 на бажану позицію

    # Відображення результату
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Приклад використання
detect_faces('test/happy_man.jpg')

# У цьому коді ми використовуємо Haar каскади для виявлення облич на зображеннях з використанням OpenCV.
# Haar каскади - це ефективний метод виявлення об'єктів, який може швидко ідентифікувати об'єкти на зображеннях,
# зокрема обличчя. Ми перетворюємо зображення в відтінки сірого для спрощення обробки, а потім використовуємо функцію
# detectMultiScale для виявлення облич. Для кожного виявленого обличчя ми малюємо рамку навколо нього.