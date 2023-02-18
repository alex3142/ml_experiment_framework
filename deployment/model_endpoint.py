import pickle
import os
from typing import (
    Dict,
    Tuple,
    Union,
)
from pathlib import Path
import json

import pandas as pd
from flask import Flask
from flask_restful import (
    Resource,
    Api,
    reqparse,
)

# this is a bit of a work around - normally these processors would be
# in their own repo which would be pip installed from git/internal
# pypi , but for time I'm cheating here a little

import sys

sources_root = Path(os.getcwd()).parent
if str(sources_root) not in sys.path:
    sys.path.append(str(sources_root))

from modelling.preprocessing.model_preprocessing_pipeline import (
    AgeProcessor,
    CarDataProcessor,
    IndexSettingProcessor,
    ColumnRemoverProcessor,
    RowOrderProcessor,
    MetaProcessor,
)


app = Flask(__name__)
api = Api(app)


class Predict(Resource):

    def __init__(self):

        modelling_folder = Path(os.path.abspath(os.path.dirname(__file__))).parent.joinpath(os.environ['MODEL_DIR'])

        model_path = modelling_folder.joinpath(
            os.environ['MODEL_LOCATION']
        ).joinpath(os.environ['MODEL_NAME'])

        preproc_path = modelling_folder.joinpath(
            os.environ['PREPROC_LOCATION']
        ).joinpath(os.environ['PREPROC_NAME'])

        self._date_col_names = ['start_date', 'cust_dob', 'end_date', 'car_age']

        self._model = pickle.load(open(model_path, 'rb'))
        self._processor = pickle.load(open(preproc_path, 'rb'))
        self._input_data_key = 'input_data'
        super().__init__()

    def _predict(self, input_data: pd.DataFrame) -> float:
        """
        Deliver prediction from model
        :param input_data: pd.DataFrame single row pandas df
        :param data_col_name: str - col name of data to be used in prediction
        :return: float prediction
        """
        # TODO - make work for batch
        class_mask = self._model.classes_ == 1
        pred_proba = self._model.predict_proba(input_data).reshape((2,))
        return self._model.predict(input_data)[0], pred_proba[class_mask][0]

    def _preprocess(self, input_data: str) -> pd.DataFrame:
        """
        run preprocessing of raw data to prepare for pipeline
        :param input_data: - str raw text to process
        :return: pd.DataFrame - processed data
        """

        df = pd.DataFrame(
            data=input_data,
            index=[0]
        )

        for col_name in self._date_col_names:
            df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d')

        return self._processor.process(df)

    def predict(self, input_data: str) -> float:
        """
        External facing prediction method to take raw text and return prediction
        :param input_text: str - raw text to precdict
        :return: float prediction
        """
        preprocessed_data = self._preprocess(input_data=input_data)
        prediction, pred_proba = self._predict(preprocessed_data)
        return int(prediction), pred_proba

    def post(self) -> Tuple[Dict[str, float], int]:
        """
        Entry point for class
        :return: Tuple[Dict[str, float], int] - prediction response
        """
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument(self._input_data_key, required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary
        input_data = json.loads(args[self._input_data_key].replace("'", '"'))
        prediction, pred_proba = self.predict(input_data)

        return {
                   'prediction': prediction,
                    'pred_proba': pred_proba,
                   'plcy_no': input_data['plcy_no'],
                   'customer_no': input_data['customer_no'],
                   'model': os.environ['MODEL_NAME'],
                   'preproc': os.environ['PREPROC_NAME']
               }, 200  # return data with 200 OK


api.add_resource(Predict, '/get_prediction')  # '/predict' is the entry point

if __name__ == '__main__':
    app.run(host=os.environ.get('host', '0.0.0.0'), port=os.environ.get('port', 5000))

    #https://abdul-the-coder.medium.com/how-to-deploy-a-containerised-python-web-application-with-docker-flask-and-uwsgi-8862a08bd5df
