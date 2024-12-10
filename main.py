import os  # Імпортуємо модуль для роботи з операційною системою
os.environ["OMP_NUM_THREADS"] = "1"  # Встановлюємо кількість потоків для OpenMP (для оптимізації)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Вимкнути паралелізм токенізатора

from multiprocessing import freeze_support  # Імпортуємо функцію для підтримки заморозки коду
import transformers  # Імпортуємо бібліотеку transformers
from transformers import MarianMTModel, MarianTokenizer  # Імпортуємо класи для моделі та токенізатора MarianMT

import torch  # Імпортуємо бібліотеку PyTorch
from langdetect import detect  # Імпортуємо функцію для розпізнавання мови

def detect_language(text):
    
    try:
        return detect(text)  # Використовуємо langdetect для визначення мови
    except:
        return "unknown"  # Повертаємо "unknown" у разі помилки

def translate_text(text, source_lang, target_lang):
    
    if source_lang == 'uk':
        model_name = 'Helsinki-NLP/opus-mt-uk-en'  # Вибираємо модель для перекладу з української на англійську
    elif source_lang == 'en':
        model_name = 'Helsinki-NLP/opus-mt-en-uk'  # Вибираємо модель для перекладу з англійської на українську
    else:
        raise ValueError("Непідтримувана мова")  # Генеруємо виняток, якщо мова не підтримується

    tokenizer = MarianTokenizer.from_pretrained(model_name)  # Завантажуємо токенізатор
    model = MarianMTModel.from_pretrained(model_name)  # Завантажуємо модель

    batch = tokenizer([text], return_tensors="pt", padding=True)  # Токенізуємо текст та готуємо його для моделі
    generated_ids = model.generate(**batch)  # Генеруємо переклад
    translated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # Декодуємо переклад
    return translated  # Повертаємо перекладений текст

if __name__ == '__main__':  # Перевіряємо, чи скрипт запущено як основний
    freeze_support()  # Викликаємо freeze_support() для підтримки заморозки

    while True:  # Запускаємо нескінченний цикл
        text = input("Введіть текст для перекладу: ")  # Отримуємо текст від користувача
        
        source_lang = detect_language(text)  # Визначаємо мову оригіналу
        print("Визначена мова оригіналу:", source_lang)  # Виводимо визначену мову

        if source_lang in ('uk', 'en'):  # Перевіряємо, чи підтримується мова
            target_lang = 'en' if source_lang == 'uk' else 'uk'  # Визначаємо мову перекладу
            try:
                translated_text = translate_text(text, source_lang, target_lang)  # Перекладаємо текст
                print("Переклад:", translated_text)  # Виводимо переклад
            except ValueError as e:  # Обробляємо виняток ValueError
                print(e)  # Виводимо повідомлення про помилку
        else:
            print("Не вдалося розпізнати мову або мова не підтримується.")  # Виводимо повідомлення про помилку