# Імпортуємо необхідні бібліотеки
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Завантажуємо датасет Iris
iris = load_iris()

# Розділяємо датасет на вхідні характеристики (X) та мітки класів (y)
X, y = iris.data, iris.target

# Розділяємо датасет на навчальну (train) та тестову (test) вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормалізуємо дані за допомогою StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Створюємо модель класифікатора KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Навчаємо модель на навчальному наборі даних
knn.fit(X_train_scaled, y_train)

# Робимо прогнози на тестовій вибірці
predictions = knn.predict(X_test_scaled)

# Виводимо звіт про класифікацію та точність моделі
print("Звіт про класифікацію:\n", classification_report(y_test, predictions))
print("Точність моделі:", accuracy_score(y_test, predictions))
