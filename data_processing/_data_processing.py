from typing import (
    Union,
    List,
    Tuple,
    Optional,
    Dict,
)
from abc import ABC
from pathlib import Path

import spacy
import pandas as pd


class TextProcessor(ABC):

    @staticmethod
    def get_processed_col_name(col_name: str):
        return col_name if col_name.endswith('_processed') else f"{col_name}_processed"

    def process(self, *args, **kwargs):
        raise NotImplemented()


class MetaProcessor(TextProcessor):

    def __init__(self, processors: Union[List[TextProcessor], Tuple[TextProcessor, ...]]) -> None:
        self._processors = processors

    def process(self, data_to_process: pd.DataFrame, col_name_to_process: str) -> pd.DataFrame:
        data_to_process_copy = data_to_process.copy()
        processed_col_name = self.get_processed_col_name(col_name_to_process)
        data_to_process_copy[processed_col_name] = data_to_process_copy[col_name_to_process].copy()
        for processor in self._processors:
            data_to_process_copy = processor.process(data_to_process_copy, processed_col_name)

        return data_to_process_copy


class LowerProcessor(TextProcessor):

    @staticmethod
    def process(data_to_process: pd.DataFrame, col_name_to_process: str) -> pd.DataFrame:
        data_to_process_copy = data_to_process.copy()
        new_col_name = LowerProcessor.get_processed_col_name(col_name_to_process)
        data_to_process_copy[new_col_name] = data_to_process_copy[col_name_to_process].str.lower()
        return data_to_process_copy


class StandardProcessor(TextProcessor):

    chunk_filename = 'processed_data_chunk_%s.csv'

    def __init__(
            self,
            language_model: str = 'en_core_web_sm',
            stopwords_update_map: Optional[Dict[str, bool]] = None,
    ) -> None:

        self._nlp = spacy.load(language_model)
        self._nlp.remove_pipe("lemmatizer")
        self._nlp.remove_pipe("ner")
        self._nlp.remove_pipe("parser")
        self._nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()

        if stopwords_update_map is not None:
            self._update_stopwords(stopwords_update_map=stopwords_update_map)

    def _update_stopwords(self, stopwords_update_map: Dict[str, bool]) -> None:

        for word, stop_bool in stopwords_update_map.items():
            self._nlp.vocab[word].is_stop = stop_bool

    def process(self, data_to_process: pd.DataFrame, col_name_to_process: str) -> pd.DataFrame:

        processed_text = []
        for doc in self._nlp.pipe(data_to_process[col_name_to_process], batch_size=1000, n_process=5):
            processed_text.append(' '.join([word.lemma_ for word in doc if not word.is_stop]))

        processed_col_name = self.get_processed_col_name(col_name_to_process)
        processed_data = pd.Series(
            data=processed_text,
            index=data_to_process.index,
            name=processed_col_name
        )

        return data_to_process.assign(**{processed_col_name: processed_data})

    def process_chunks(self, data_to_process: pd.DataFrame, col_name_to_process: str, n_chunks: int) -> None:

        data_chunked = [data_to_process[i:i + n_chunks] for i in range(0, data_to_process.shape[0], n_chunks)]

        for chunk_index, chunk in enumerate(data_chunked):
            if Path(self.chunk_filename % chunk_index).exists():
                print(f"chunk {chunk_index} already exists")
                continue
            print(f"chunk index = {chunk_index}")
            chunk = self.process(chunk, col_name_to_process)
            chunk.to_csv(self.chunk_filename % chunk_index)

    @staticmethod
    def combine_chunks(filename: str) -> None:

        chunk_index = 0
        chunked_data = []
        while Path(StandardProcessor.chunk_filename % chunk_index).exists():
            print("opening file ", StandardProcessor.chunk_filename % chunk_index)
            chunked_data.append(pd.read_csv(StandardProcessor.chunk_filename % chunk_index, index_col=False))
            chunk_index += 1

        all_data = pd.concat(chunked_data)
        all_data.to_csv(filename, index=False)
