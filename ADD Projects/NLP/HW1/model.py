from typing import List

class PrefixTreeNode:
    def __init__(self):
        # словарь с буквами, которые могут идти после данной вершины
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False


class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        """
        vocabulary: список всех уникальных токенов в корпусе
        """
        self.root = PrefixTreeNode()

        for word in vocabulary:
            self.add_word(word)

    def add_word(self, word: str):
        """
        Добавление слова в префиксное дерево.
        """
        current_node = self.root

        for char in word:
            if char not in current_node.children:
                current_node.children[char] = PrefixTreeNode()
            current_node = current_node.children[char]

        current_node.is_end_of_word = True

    def search_prefix(self, prefix) -> List[str]:
        """
        Возвращает все слова, начинающиеся на prefix
        prefix: str – префикс слова
        """

        current_node = self.root

        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # если отсутствует в дереве

        # возвращаем все слова с данного узла
        return self._collect_words(prefix, current_node)

    def _collect_words(self, prefix: str, node: PrefixTreeNode) -> List[str]:
        """
        Рекурсивно собирает все слова, начинающиеся с данного узла.
        """
        words = []
        if node.is_end_of_word:
            words.append(prefix)

        for char, child_node in node.children.items():
            words.extend(self._collect_words(prefix + char, child_node))

        return words


import math
from collections import Counter
from typing import List, Tuple


class WordCompletor:
    def __init__(self, corpus):
        """
        corpus: list – корпус текстов
        """
        flattened_corpus = [word for sublist in corpus for word in sublist]

        # частотность слов
        self.word_counts = Counter(flattened_corpus)

        # общее кол-во слов
        self.total_words = sum(self.word_counts.values())

        self.prefix_tree = PrefixTree(list(self.word_counts.keys()))

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        """
        Возвращает список слов, начинающихся на prefix,
        с их вероятностями (нормировать ничего не нужно)
        """
        # находим слова по префиксу
        words = self.prefix_tree.search_prefix(prefix)

        # считаем вероятность
        probs = [self.word_counts[word] / self.total_words for word in words]

        return words, probs


from typing import List, Tuple
from collections import Counter, defaultdict
from itertools import islice


class NGramLanguageModel:
    def __init__(self, corpus, n):
        """
        Инициализация n-граммной модели.
        corpus: список списков слов (корпус)
        n: длина n-грамм
        """
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        for sentence in corpus:
            for ngram_size in range(1, len(sentence) + 1):  # генерация нграмм разной длины
                for start_idx in range(len(sentence) - ngram_size + 1):  # перебор индексов
                    ngram = tuple(sentence[start_idx:start_idx + ngram_size])
                    self.ngram_counts[ngram] += 1
                    if len(ngram) > 1:  # учитываем только нграммы длиной > 1
                        context_key = ngram[:-1]
                        self.context_counts[context_key] += 1

    def get_next_words_and_probs(self, prefix: list) -> (List[str], List[float]):
        """
        Возвращает список слов, которые могут идти после prefix,
        а так же список вероятностей этих слов
        """

        prefix_tuple = tuple(prefix)
        context_count = self.context_counts.get(prefix_tuple, 0)
        if context_count == 0:
            return [], []

        next_word_candidates = {}
        for ngram, frequency in self.ngram_counts.items():
            if len(ngram) == len(prefix_tuple) + 1 and ngram[:-1] == prefix_tuple:
                following_word = ngram[-1]
                next_word_candidates[following_word] = frequency

        next_words = []
        probs = []
        for candidate, freq in next_word_candidates.items():
            probability = freq / context_count
            next_words.append(candidate)
            probs.append(probability)

        return next_words, probs



from typing import Union


class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, text: Union[str, list], n_words=3, n_texts=1) -> list[list[str]]:
        """
        Возвращает возможные варианты продолжения текста (по умолчанию только один)

        text: строка или список слов – написанный пользователем текст
        n_words: число слов, которые дописывает n-граммная модель
        n_texts: число возвращаемых продолжений (пока что только одно)

        return: list[list[srt]] – список из n_texts списков слов, по 1 + n_words слов в каждом
        Первое слово – это то, которое WordCompletor дополнил до целого.
        """

        if isinstance(text, str):
            text = text.strip().split()  # токенезируем текст
        else:
            text = text[:]

        if not text:  # если нет текста, возвращаем пустой список
            return []

        suggestions = []

        for _ in range(n_texts):  # смотрим только один вариант продолжения
            current_text = text[:]
            last_word = current_text[-1]

            # выбираем последнее слово
            completions, probs = self.word_completor.get_words_and_probs(last_word)
            if completions:
                max_prob_index = probs.index(max(probs))
                current_text[-1] = completions[max_prob_index]

            # задаем контекст
            suggestion = [current_text[-1]]
            context = current_text[-(self.n_gram_model.n - 1):] if self.n_gram_model.n > 1 else []

            # генерим следующие слова
            for _ in range(n_words):
                next_words, next_probs = self.n_gram_model.get_next_words_and_probs(context)
                if not next_words:
                    break  # если нет предиктов то останавливаем цикл
                max_prob_index = next_probs.index(max(next_probs))
                next_word = next_words[max_prob_index]
                suggestion.append(next_word)

                # обновляем контекст
                if self.n_gram_model.n > 1:
                    context = context[1:] + [next_word] if len(context) >= self.n_gram_model.n - 1 else context + [
                        next_word]
                else:
                    context = [next_word]

            suggestions.append(suggestion)

        return suggestions



