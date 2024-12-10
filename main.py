import os
# Встановлюємо змінні оточення для обмеження кількості потоків та 
# відключення паралелізму токенізатора для уникнення конфліктів.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from multiprocessing import freeze_support  # Для сумісності з Windows
import transformers  # Бібліотека для роботи з моделями NLP
from transformers import MarianMTModel, MarianTokenizer  # Модель та токенізатор для машинного перекладу

import torch  # Бібліотека для роботи з тензорами

def translate_text(text, source_lang, target_lang):
    """
    Перекладає текст з однієї мови на іншу за допомогою моделі MarianMT.

    Args:
        text: Текст для перекладу.
        source_lang: Код мови оригіналу ('uk' або 'en').
        target_lang: Код мови перекладу ('en' або 'uk').

    Returns:
        Перекладений текст.
    """
    # Вибір моделі перекладу в залежності від мовної пари
    if source_lang == 'uk':
        model_name = 'Helsinki-NLP/opus-mt-uk-en'  # Модель для перекладу з української на англійську
    elif source_lang == 'en':
        model_name = 'Helsinki-NLP/opus-mt-en-uk'  # Модель для перекладу з англійської на українську
    else:
        raise ValueError("Непідтримувана мова")  # Помилка, якщо мова не підтримується

    # Завантаження токенізатора та моделі
    tokenizer = MarianTokenizer.from_pretrained(model_name)  
    model = MarianMTModel.from_pretrained(model_name)

    # Токенізація тексту та перетворення в тензор
    batch = tokenizer([text], return_tensors="pt", padding=True)  
    # Генерація перекладу
    generated_ids = model.generate(**batch)  
    # Декодування результату
    translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  
    return translated

if __name__ == '__main__':
    freeze_support()  # Для сумісності з Windows

    while True:  # Нескінченний цикл для перекладу тексту
        text = input("Введіть текст для перекладу: ")
        source_lang = input("Введіть мову оригіналу (uk або en): ")
        target_lang = input("Введіть мову перекладу (en або uk): ")

        try:
            translated_text = translate_text(text, source_lang, target_lang)  # Виклик функції перекладу
            print("Переклад:", translated_text)  # Виведення перекладеного тексту
        except ValueError as e:  # Обробка помилки непідтримуваної мови
            print(e)