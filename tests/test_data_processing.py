import datetime

import pandas as pd
import pytest

from data_processing import (
    StandardProcessor,
    LowerProcessor,
    MetaProcessor,
    ColumnRemoverProcessor,
    IndexSettingProcessor,
    AgeProcessor,
    CarDataProcessor,
    RowOrderProcessor,
)


def test_StandardProcessor():

    test_case = pd.DataFrame(
            {
            'col_1': [1, 2, 3],
            'col_2': ['HELLO world', 'we the', 'london and manchester']
            }
        )

    expected_result = pd.DataFrame(
        {
        'col_1': [1, 2, 3],
        'col_2': ['HELLO world', 'we the', 'london and manchester'],
        'col_2_processed': ['HELLO world', '', 'london manchester']
        }
    )
    assert expected_result.equals(StandardProcessor().process(test_case, 'col_2'))


def test_LowerProcessor():
    test_case = pd.DataFrame(
        {
        'col_1': [1, 2, 3],
        'col_2': ['HELLO world', 'we THE', 'London and Manchester']
        }
    )

    expected_result = pd.DataFrame(
        {
        'col_1': [1, 2, 3],
        'col_2': ['HELLO world', 'we THE', 'London and Manchester'],
        'col_2_processed': ['hello world', 'we the', 'london and manchester']
        }
    )
    assert expected_result.equals(LowerProcessor.process(test_case, 'col_2'))


def test_AgeProcessor___calculate_age():
    proc = AgeProcessor(
        lower_date_col_name='start_date',
        upper_date_col_name='end_date',
        new_col_name='new_col_name',
        as_years=False,
    )

    test_case = pd.DataFrame(
        {
            'start_date': [
                datetime.datetime(year=2020, month=4, day=1),
                datetime.datetime(year=2004, month=5, day=1),
            ],
            'end_date': [
                datetime.datetime(year=2020, month=4, day=2),
                datetime.datetime(year=2004, month=6, day=1),
            ],
        }
    )

    expected_results = pd.Series([2.0, 32.0])

    pd.testing.assert_series_equal(
        expected_results, proc._calculate_age(
            date_lower=test_case['start_date'],
            date_upper=test_case['end_date']
        )
    )


def test_AgeProcessor___process(mocker):
    proc = AgeProcessor(
        lower_date_col_name='start_date',
        upper_date_col_name='end_date',
        new_col_name='new_col',
        as_years=False,
    )

    calculate_age_mock_return = pd.Series([2.0, 33.0])
    calculate_age_mock = mocker.MagicMock(
        return_value=calculate_age_mock_return
    )

    proc._calculate_age = calculate_age_mock

    test_case = pd.DataFrame(
        {
            'col_1': ['a', 'b'],
            'start_date': [
                datetime.datetime(year=2020, month=4, day=1),
                datetime.datetime(year=2004, month=5, day=1),
            ],
            'end_date': [
                datetime.datetime(year=2020, month=4, day=2),
                datetime.datetime(year=2004, month=6, day=1),
            ],
        }
    )

    expected_results = pd.DataFrame(
        {
            'col_1': ['a', 'b'],
            'start_date': [
                datetime.datetime(year=2020, month=4, day=1),
                datetime.datetime(year=2004, month=5, day=1),
            ],
            'end_date': [
                datetime.datetime(year=2020, month=4, day=2),
                datetime.datetime(year=2004, month=6, day=1),
            ],
            'new_col__processed': [2.0, 33.0],
        }
    )

    actual_results = proc._process(
            data_to_process=test_case
        )

    assert calculate_age_mock.call_count == 1
    assert len(calculate_age_mock.call_args_list) == 1
    pd.testing.assert_series_equal(
        calculate_age_mock.call_args_list[0].kwargs['date_lower'],
        test_case['start_date']
    )
    pd.testing.assert_series_equal(
        calculate_age_mock.call_args_list[0].kwargs['date_upper'],
        test_case['end_date']
    )

    pd.testing.assert_frame_equal(
        expected_results, actual_results
    )


def test_AgeProcessor__process(mocker):
    proc = AgeProcessor(
        lower_date_col_name='start_date',
        upper_date_col_name='end_date',
        new_col_name='new_col',
        as_years=False,
    )

    _process_mock_return = '_process_mock_return_'
    _process_mock = mocker.MagicMock(
        return_value=_process_mock_return
    )
    proc._process = _process_mock

    calling_kwargs = {'input_1': 'input_1_value'}

    actual_results = proc.process(**calling_kwargs)

    _process_mock.assert_called_once_with(**calling_kwargs)
    assert _process_mock_return == actual_results


def test_CarDataProcessor___remove_incorrect_data(mocker):
    proc = CarDataProcessor(
        lower_date_col_name='start_date',
        upper_date_col_name='end_date',
        new_col_name='new_col',
        car_col_names=['col_2', 'col_3'],
        as_years=False,
    )

    calculate_age_mock_return = pd.Series([2.0, 33.0])
    calculate_age_mock = mocker.MagicMock(
        return_value=calculate_age_mock_return
    )

    proc._calculate_age = calculate_age_mock

    test_case = pd.DataFrame(
        {
            'col_1': ['a', 'b'],
            'new_col__processed': [2.0, -5],
            'col_2': ['x', 'y'],
            'col_3': [5, 6],
        }
    )

    expected_results = pd.DataFrame(
        {
            'col_1': ['a', 'b'],
            'new_col__processed': [2.0, float('nan')],
            'col_2': ['x', float('nan')],
            'col_3': [5, float('nan')],
        }
    )

    actual_results = proc._remove_incorrect_data(
            data_to_process=test_case
        )

    pd.testing.assert_frame_equal(
        expected_results.sort_index(axis=1),
        actual_results.sort_index(axis=1),
    )


def test_CarDataProcessor__process(mocker):
    proc = CarDataProcessor(
        lower_date_col_name='start_date',
        upper_date_col_name='end_date',
        new_col_name='new_col',
        car_col_names=['col_2', 'col_3'],
        as_years=False,
    )

    test_case = 'input_arg'

    _process_mock_return = '_process_mock_return_'
    _process_mock = mocker.MagicMock(
        return_value=_process_mock_return
    )
    proc._process = _process_mock

    _remove_incorrect_data_mock_return = '_remove_incorrect_data_mock_return_'
    _remove_incorrect_data_mock = mocker.MagicMock(
        return_value=_remove_incorrect_data_mock_return
    )
    proc._remove_incorrect_data = _remove_incorrect_data_mock

    actual_result = proc.process(data_to_process=test_case)

    _process_mock.assert_called_once_with(
        **{'data_to_process': test_case}
    )
    _remove_incorrect_data_mock.assert_called_once_with(
        **{'data_to_process': _process_mock_return}
    )

    assert _remove_incorrect_data_mock_return == actual_result


def test_ColumnRemoverProcessor():
    proc = ColumnRemoverProcessor(col_names_to_remove=['col_1', 'col_2'])

    test_case = pd.DataFrame(
        {
            'col_1': [0, 1, 0],
            'col_2': [1, 1, 0],
            'col_3': [1, 1, 1],
        }
    )

    expected_result = pd.DataFrame(
        {
            'col_3': [1, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(expected_result, proc.process(test_case))


def test_RowOrderProcessor():
    proc = RowOrderProcessor(col_names_to_order_by=['end_date'])

    test_case = pd.DataFrame(
        {
            'col_1': [1, 2, 4],
            'end_date': [
                datetime.datetime(year=2021, month=4, day=2),
                datetime.datetime(year=2020, month=4, day=2),
                datetime.datetime(year=2025, month=4, day=2),
            ]
        },
        index=['a', 'b', 'c']
    )

    expected_result = pd.DataFrame(
        {
            'col_1': [2, 1, 4],
            'end_date': [
                datetime.datetime(year=2020, month=4, day=2),
                datetime.datetime(year=2021, month=4, day=2),
                datetime.datetime(year=2025, month=4, day=2),
            ]
        },
        index=['b', 'a', 'c']
    )

    pd.testing.assert_frame_equal(
        expected_result, proc.process(data_to_process=test_case)
    )


def test_IndexSettingProcessor():
    proc = IndexSettingProcessor(col_names_for_index=['col_1', 'col_2'])

    test_case = pd.DataFrame(
        {
            'col_1': ['a', 'b', 'c'],
            'col_2': [1, 1, 0],
            'col_3': [1, 1, 1],
        }

    )

    index_data = [
        ['a', 'b', 'c'],
        [1, 1, 0]
    ]

    index_names = ('col_1', 'col_2')

    expected_result = pd.DataFrame(
        {
            'col_3': [1, 1, 1],
        },
        index=pd.MultiIndex.from_arrays(index_data, names=index_names)
    )

    pd.testing.assert_frame_equal(expected_result, proc.process(test_case))


def test_MetaProcessor(mocker):
    input_mock_return = 'input_mock_return'
    input_mock = mocker.MagicMock()
    input_mock.copy = mocker.MagicMock(return_value=input_mock_return)

    mock_1_proc_return = 'mock_1_proc_return'
    proc_1_mock = mocker.MagicMock()
    proc_1_mock.process = mocker.MagicMock(return_value=mock_1_proc_return)

    mock_2_proc_return = 'mock_2_proc_return'
    proc_2_mock = mocker.MagicMock()
    proc_2_mock.process = mocker.MagicMock(return_value=mock_2_proc_return)

    proc = MetaProcessor(processors=(proc_1_mock, proc_2_mock))

    actual_output = proc.process(input_mock)

    proc_1_mock.process.assert_called_once_with(
        **{'data_to_process': input_mock_return})
    proc_2_mock.process.assert_called_once_with(
        **{'data_to_process': mock_1_proc_return})
    assert actual_output == mock_2_proc_return
