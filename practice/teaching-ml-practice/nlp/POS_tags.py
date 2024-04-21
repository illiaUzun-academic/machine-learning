import nltk
from nltk.tokenize import word_tokenize

# Текст для аналізу
text = "Artificial intelligence is transforming the world."

# Токенізація тексту
tokens = word_tokenize(text)

# Визначення частин мови для кожного слова
tagged_tokens = nltk.pos_tag(tokens)

print("Теги частин мови:")
for token, tag in tagged_tokens:
    print(f"{token}: {tag}")

# Цей код використовує NLTK для токенізації тексту і визначення частин мови кожного слова.
# Теги частин мови допомагають розуміти граматичну структуру речення і можуть бути використані
# для подальшого лінгвістичного аналізу або обробки тексту.

# • Verbs: VB, VBP, VBZ, VBD, VBG, VBN
# – base, present-non-3rd, present-3rd, past, -ing, -en
# • Nouns: NNP, NNPS, NN, NNS
# – proper/common, singular/plural (singular includes mass + generic)
# • Adjectives: JJ, JJR, JJS (base, comparative, superlative)
# • Adverbs: RB, RBR, RBS, RP (base, comparative, superlative, particle)
# • Pronouns: PRP, PP$ (personal, possessive)
# • Interogatives: WP, WP$, WDT, WRB (compare to: PRP, PP$, DT, RB)
# • Other Closed Class: CC, CD, DT, PDT, IN, MD
# • Punctuation: # $ . , : ( ) “ ” '' ' `
# • Weird Cases: FW(deja vu), SYM (@), LS (1, 2, a, b), TO (to), POS('s, '),
#   UH (no, OK, well), EX (it/there
