import pandas as pd
import pytest

from data_processing import (
    StandardProcessor,
    LowerProcessor,
    MetaProcessor,
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


def test_MetaProcessor():
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
        'col_2_processed': ['hello world', '', 'london manchester']
        }
    )

    proc = MetaProcessor(processors=(LowerProcessor(), StandardProcessor()))
    assert expected_result.equals(proc.process(test_case, 'col_2'))


