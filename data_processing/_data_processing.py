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


class DataProcessor(ABC):

    _processed_suffix = '__processed'

    @staticmethod
    def get_processed_col_name(col_name: str):
        return col_name if col_name.endswith('_processed') else f"{col_name}_processed"

    def process(self, *args, **kwargs):
        raise NotImplemented()


class MetaProcessor(DataProcessor):

    def __init__(self, processors: Union[List[TextProcessor], Tuple[TextProcessor, ...]]) -> None:
        self._processors = processors

    def process(self, data_to_process: pd.DataFrame, col_name_to_process: str) -> pd.DataFrame:
        data_to_process_copy = data_to_process.copy()
        processed_col_name = self.get_processed_col_name(col_name_to_process)
        data_to_process_copy[processed_col_name] = data_to_process_copy[col_name_to_process].copy()
        for processor in self._processors:
            data_to_process_copy = processor.process(data_to_process_copy, processed_col_name)

        return data_to_process_copy


class LowerProcessor(DataProcessor):

    @staticmethod
    def process(data_to_process: pd.DataFrame, col_name_to_process: str) -> pd.DataFrame:
        data_to_process_copy = data_to_process.copy()
        new_col_name = LowerProcessor.get_processed_col_name(col_name_to_process)
        data_to_process_copy[new_col_name] = data_to_process_copy[col_name_to_process].str.lower()
        return data_to_process_copy


class StandardProcessor(DataProcessor):

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


class AgeProcessor(DataProcessor):

    def __init__(
            self,
            lower_date_col_name: str,
            upper_date_col_name: str,
            new_col_name: str,
            as_years: bool = True,
            days_in_year: float = 365.25
    ) -> None:

        self._lower_date_col_name = lower_date_col_name
        self._upper_date_col_name = upper_date_col_name
        self._new_col_name = new_col_name
        self._as_years = as_years
        self._days_in_year = days_in_year

    def _calculate_age(
            self,
            date_lower: pd.Series,
            date_upper: pd.Series
        ) -> pd.Series:
        """

        :param self:
        :param date_lower:
        :param date_upper:
        :return:
        """

        # plus one needed to make age inclusive
        denominator = self._days_in_year if self._as_years else 1
        return (
                (date_upper - date_lower).dt.days + 1
               ) / denominator

    def _process(self, data_to_process: pd.DataFrame) -> pd.DataFrame:
        data_to_process_copy = data_to_process.copy()

        new_date_data = self._calculate_age(
            date_lower=data_to_process[self._lower_date_col_name],
            date_upper=data_to_process[self._upper_date_col_name],
        )

        new_date_data = new_date_data.rename(
            f'{self._new_col_name}{self._processed_suffix}'
        )
        data_to_process_copy = data_to_process_copy.assign(
            **{new_date_data.name: new_date_data}
        )

        return data_to_process_copy

    def process(self, *args, **kwargs) -> pd.DataFrame:
        return self._process(*args, **kwargs)


class CarDataProcessor(AgeProcessor):

    def __init__(
            self,
            lower_date_col_name: str,
            upper_date_col_name: str,
            new_col_name: str,
            car_col_names: List[str],
            as_years: bool = True,
            days_in_year: float = 365.25,

    ) -> None:
        super().__init__(
            lower_date_col_name=lower_date_col_name,
            upper_date_col_name=upper_date_col_name,
            new_col_name=new_col_name,
            as_years=as_years,
            days_in_year=days_in_year,
        )

        self._car_col_names = car_col_names

    def _remove_incorrect_data(
            self,
            data_to_process: pd.DataFrame
    ) -> pd.DataFrame:
        """
        TODO - not sure this is the best way of doing this, but couldn't
        find a better solution - more research may be needed
        :param data_to_process:
        :return:
        """

        acceptable_age_mask = data_to_process[
                   f'{self._new_col_name}{self._processed_suffix}'
               ] >= 0

        affected_col_names = self._car_col_names.copy()
        affected_col_names.append(
            f'{self._new_col_name}{self._processed_suffix}'
        )

        unaffected_cols_df = data_to_process.drop(columns=affected_col_names)

        new_cols = [unaffected_cols_df]
        for col in affected_col_names:
            new_cols.append(
                data_to_process[col].where(acceptable_age_mask, float('nan'))
            )

        return pd.concat(new_cols, axis=1)

    def process(self, data_to_process: pd.DataFrame) -> pd.DataFrame:

        data_with_age = self._process(data_to_process=data_to_process)
        return self._remove_incorrect_data(data_to_process=data_with_age)


class IndexSettingProcessor(DataProcessor):

    def __init__(self, col_names_for_index: Union[List[str], str]) -> None:
        self._col_names_for_index = col_names_for_index

    def process(self, data_to_process: pd.DataFrame) -> pd.DataFrame:
        return data_to_process.set_index(
            keys=self._col_names_for_index
        )


class ColumnRemoverProcessor(DataProcessor):

    def __init__(self, col_names_to_remove: List[str]) -> None:
        self._col_names = col_names_to_remove

    def process(self, data_to_process: pd.DataFrame) -> pd.DataFrame:
        return data_to_process.drop(columns=self._col_names)


class RowOrderProcessor(DataProcessor):

    def __init__(
            self,
            col_names_to_order_by: List[str],
            ascending: bool = True,
    ) -> None:
        self._col_names_to_order_by = col_names_to_order_by
        self._ascending = ascending

    def process(self, data_to_process: pd.DataFrame) -> pd.DataFrame:
        return data_to_process.sort_values(
            by=self._col_names_to_order_by,
            ascending=self._ascending
        )




