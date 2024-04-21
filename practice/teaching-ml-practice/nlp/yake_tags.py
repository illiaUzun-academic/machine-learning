import yake

# Текст для аналізу
text = """
Climate change is one of the most pressing issues facing the world today. 
It impacts every continent, affecting agriculture, human health, ecosystems, water supplies, and even geopolitics. 
Cutting down on greenhouse gas emissions is crucial to mitigating the worst effects of climate change. 
Renewable energy sources like solar and wind power play a significant role in reducing our reliance on fossil fuels.
"""

# Налаштування yake
kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=5, features=None)

# Визначення ключових слів
keywords = kw_extractor.extract_keywords(text)

print("Ключові слова:")
for kw in keywords:
    print(kw[0])

# Цей код використовує бібліотеку yake для визначення ключових слів у тексті.
# Це може бути особливо корисно для автоматичної обробки та аналізу великих обсягів тексту,
# допомагаючи швидко ідентифікувати основні теми та концепції.
