import re


# Функція для перетворення запитання на SQL-запит
def question_to_sql(question):
    # Шаблони для пошуку в тексті запитання
    pattern_select = re.compile(r'як(і|а|е|у)?\s+(.+?)\s+у\s+(.+)', re.IGNORECASE)
    pattern_count = re.compile(r'скільки\s+(.+?)\s+у\s+(.+)', re.IGNORECASE)

    # Спроба знайти відповідності для SELECT запитів
    match_select = pattern_select.search(question)
    if match_select:
        columns, table = match_select.groups()[1], match_select.groups()[2]
        return f"SELECT {columns} FROM {table};"

    # Спроба знайти відповідності для COUNT запитів
    match_count = pattern_count.search(question)
    if match_count:
        condition, table = match_count.groups()[0], match_count.groups()[1]
        return f"SELECT COUNT({condition}) FROM {table};"

    return "Не зміг перетворити запитання на SQL-запит."


# Приклад запитання
question1 = "Які імена у таблиці співробітників?"
question2 = "Скільки продуктів у таблиці товарів?"

# Перетворення запитань на SQL-запити
sql_query1 = question_to_sql(question1)
sql_query2 = question_to_sql(question2)

print(sql_query1)
print(sql_query2)

# Цей код ілюструє базовий метод перетворення вопросів на SQL-запити, використовуючи регулярні вирази.
# Він може бути розширений та адаптований для більш складних вопросів та більшої кількості типів запитів.
