# -*- coding: utf-8 -*-
import os
from pandas import DataFrame
from pydantic import BaseModel, field_validator, Field
import pandas as pd
from ds_project.constants import TITANICSchemaCSV


class TitanicTrainingData(BaseModel):
    """
    Schema for the Titanic dataset with data type and validation rules
    """

    df: DataFrame = Field(
        ...,
        description="Titanic dataset with required columns",
        validate_all=True,  # Validate all columns
    )

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def from_file(path_layout: os.PathLike) -> "TitanicTrainingData":
        with open(path_layout, "rb") as csv_file:
            result = pd.read_csv(csv_file, sep=",")
            return TitanicTrainingData(df=result)

    @field_validator("df")
    def data_no_empty(cls, df):
        assert not df.empty, "empty dataframe"
        return df

    @field_validator("df")
    def data_correct_cols(cls, df):
        REQUIRED_COLS = sorted(
            TITANICSchemaCSV.get_features() + TITANICSchemaCSV.get_target()
        )

        assert set(df.columns.tolist()) == set(REQUIRED_COLS), "invalid columns"
        return df

    @field_validator("df")
    def data_no_duplicates(cls, df):
        should_be_index = [
            TITANICSchemaCSV.PASSENGER_ID,
        ]

        assert df.duplicated().sum() == 0, "duplicated data"
        return df.set_index("PassengerId")


class TitanicTestingData(BaseModel):
    """
    Schema for the Titanic dataset with data type and validation rules
    """

    df: DataFrame = Field(
        ...,
        description="Titanic dataset with required columns",
        validate_all=True,  # Validate all columns
    )

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def from_file(path_layout: os.PathLike) -> "TitanicTestingData":
        with open(path_layout, "rb") as csv_file:
            result = pd.read_csv(csv_file, sep=",")
            return TitanicTestingData(df=result)

    @field_validator("df")
    def data_no_empty(cls, df):
        assert not df.empty, "empty dataframe"
        return df

    @field_validator("df")
    def data_correct_cols(cls, df):
        REQUIRED_COLS = sorted(TITANICSchemaCSV.get_features())

        assert set(df.columns.tolist()) == set(REQUIRED_COLS), "invalid columns"
        return df

    @field_validator("df")
    def data_no_duplicates(cls, df):
        should_be_index = [
            TITANICSchemaCSV.PASSENGER_ID,
        ]

        assert df.duplicated().sum() == 0, "duplicated data"
        return df.set_index("PassengerId")
