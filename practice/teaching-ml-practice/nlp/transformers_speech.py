import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Завантаження аудіофайлу
audio_file = "harvard.wav"
audio_input, sampling_rate = librosa.load(audio_file, sr=16000)

# Ініціалізація токенайзера та моделі
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Обробка аудіофайлу
input_values = tokenizer(audio_input, return_tensors="pt").input_values

# Распізнавання речі з аудіо
with torch.no_grad():
    logits = model(input_values).logits

# Декодування результатів
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

print("Транскрипція аудіо:")
print(transcription)

# Цей код демонструє, як можна використовувати модель Wav2Vec2 для преобразования речи из аудиофайла в текст.