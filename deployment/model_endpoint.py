import pickle
import os
from typing import (
    Dict,
    Tuple,
)

import pandas as pd
from flask import Flask
from flask_restful import (
    Resource,
    Api,
    reqparse,
)

from data_processing import (
    MetaProcessor,
    LowerProcessor,
    StandardProcessor,
)

app = Flask(__name__)
api = Api(app)


class Predict(Resource):

    def __init__(self):

        self._model = pickle.load(open(os.environ['model_path'], 'rb'))
        # TODO - create processor factory to create processors from environment variable/config
        self._processor = MetaProcessor(
            processors=(
                LowerProcessor(),
                StandardProcessor(
                    stopwords_update_map={'not': False, "n't": False, ',': True, '.': True, '(': True, ')': True}
                )
            )
        )
        self._text_key = 'review_full'
        super().__init__()

    def _predict(self, input_data: pd.DataFrame, data_col_name: str) -> float:
        """
        Deliver prediction from model
        :param input_data: pd.DataFrame single row pandas df
        :param data_col_name: str - col name of data to be used in prediction
        :return: float prediction
        """
        # TODO - make work for batch
        return self._model.predict(input_data[data_col_name])[0]

    def _preprocess(self, input_data: str) -> pd.DataFrame:
        """
        run preprocessing of raw data to prepare for pipeline
        :param input_data: - str raw text to process
        :return: pd.DataFrame - processed data
        """
        return self._processor.process(
            data_to_process=pd.DataFrame({self._text_key: [input_data]}, index=[0]),
            col_name_to_process=self._text_key
        )

    def predict(self, input_text: str) -> float:
        """
        External facing prediction method to take raw text and return prediction
        :param input_text: str - raw text to precdict
        :return: float prediction
        """
        print(input_text)
        preprocessed_text = self._preprocess(input_data=input_text)
        print(preprocessed_text)
        return self._predict(preprocessed_text, self._processor.get_processed_col_name(self._text_key))

    def post(self) -> Tuple[Dict[str, float], int]:
        """
        Entry point for class
        :return: Tuple[Dict[str, float], int] - prediction response
        """
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument(self._text_key, required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary
        return {'prediction': self.predict(args[self._text_key])}, 200  # return data with 200 OK


api.add_resource(Predict, '/predict')  # '/predict' is the entry point

if __name__ == '__main__':
    app.run(host=os.environ.get('host', '127.0.0.1'), port=os.environ.get('port', 5000))
