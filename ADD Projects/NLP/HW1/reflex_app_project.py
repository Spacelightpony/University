import reflex as rx
import pandas as pd
from .model import WordCompletor, NGramLanguageModel, TextSuggestion
from pathlib import Path

#путь к файлу
script_dir = Path(__file__).parent
corpus_path = script_dir / 'corpus.xlsx'


#загрузка екселя с корпусом данных
df = pd.read_excel(corpus_path)

sentences = df['updated_fully_cleaned_message'].dropna().tolist()
#токенизация корпуса
corpus = [sentence.strip().split() for sentence in sentences]

#создаем экземпляры классов
word_completor = WordCompletor(corpus)
n_gram_model = NGramLanguageModel(corpus, n=2)
text_suggester = TextSuggestion(word_completor, n_gram_model)

class State(rx.State):
    """Состояние приложения."""

    prompt: str = ""
    generated_text: list = []
    processing: bool = False
    complete: bool = False

    def generate_text(self):
        """Генерация текста на основе пользовательского ввода."""
        if self.prompt.strip() == "":
            return rx.window_alert("Поле ввода не должно быть пустым")

        self.processing, self.complete = True, False
        yield  

        #генерация текста
        try:
            suggestions = text_suggester.suggest_text(self.prompt, n_words=3, n_texts=1)
            if suggestions:
                generated_sentences = [' '.join(suggestion) for suggestion in suggestions]
                self.generated_text = generated_sentences
            else:
                self.generated_text = ["Нет доступных предложений."]
        except Exception as e:
            self.generated_text = [f"Ошибка при генерации текста: {str(e)}"]

        self.processing, self.complete = False, True

def generated_texts_box(text: str):
    """Функция для отображения сгенерированного текста."""
    return rx.box(rx.text(text))

def simple_foreach():
    """Функция для отображения списка сгенерированных текстов."""
    return rx.grid(
        rx.foreach(State.generated_text, generated_texts_box),
    )

def index():
    """Главная страница приложения."""
    return rx.center(
        rx.vstack(
            rx.heading("Генератор текста", font_size="1.5em"),
            rx.input(
                placeholder="Введите текст...",
                on_blur=State.set_prompt,
                width="25em",
            ),
            rx.button(
                "Сгенерировать текст",
                on_click=State.generate_text,
                width="25em",
                loading=State.processing
            ),
            rx.cond(
                State.complete,
                simple_foreach()
            ),
            align_items="center",
        ),
        width="100%",
        height="100vh",
    )

#создание приложения
app = rx.App()
app.add_page(index, title="Генератор текста")

