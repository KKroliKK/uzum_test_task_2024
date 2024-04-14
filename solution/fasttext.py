from typing import List

import fasttext
import numpy as np
from tqdm import tqdm

from solution.constants import LID_176_MODEL_PATH, FASTTEXT_RU_PATH, FASTTEXT_UZ_PATH


class LanguagePredictor:
    def __init__(self, model_path: str = LID_176_MODEL_PATH) -> None:
        self.model = fasttext.load_model(model_path)

    def predict_language(self, text: str) -> str:
        predicted_language = self.model.predict(text, k=1)
        language_code = predicted_language[0][0].split("__label__")[1]

        return language_code


class SentenceTokenizer:
    def __init__(self, model_path: str) -> None:
        self.model = fasttext.load_model(model_path)

    def get_sentence_embedding(self, sentence: str) -> List[float]:
        sentence_embedding = self.model.get_sentence_vector(sentence)

        return sentence_embedding


class UzSentenceTokenizer(SentenceTokenizer):
    def __init__(self, model_path: str = FASTTEXT_UZ_PATH) -> None:
        super().__init__(model_path)


class RuSentenceTokenizer(SentenceTokenizer):
    def __init__(self, model_path: str = FASTTEXT_RU_PATH) -> None:
        super().__init__(model_path)


def embed_sentences(sentences: List[str],
    language_predictor = LanguagePredictor(),
    uz_tokenizer = UzSentenceTokenizer(),
    ru_tokenizer = RuSentenceTokenizer(),
    ) -> np.ndarray:
    def embed_sentence(
        sentence: str,
    ) -> List[float]:
        language = language_predictor.predict_language(text=sentence)

        if language == "ru":
            embedding = ru_tokenizer.get_sentence_embedding(sentence=sentence)
        else:
            embedding = uz_tokenizer.get_sentence_embedding(sentence=sentence)

        return embedding

    embeddings = []

    for sentence in tqdm(sentences):
        one_line_text = sentence.replace("\n", " ")
        embedding = embed_sentence(one_line_text)
        embeddings.append(embedding)
 
    embeddings = np.array(embeddings)

    return embeddings
