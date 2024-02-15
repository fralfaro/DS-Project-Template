from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Titanic dataset by performing the following steps:

    1. Removes unnecessary columns: 'Name' and 'Ticket'.
    2. Fills missing values in the 'Age' column with the average age.
    3. Converts 'Pclass', 'SibSp', and 'Parch' columns to string data types.

    Args:
        df (pd.DataFrame): The Titanic dataset to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Drop irrelevant columns:
    df = df.drop(columns=["Name", "Ticket"],axis=1)

    # Handle missing values in Age:
    average_age = round(df["Age"].mean())
    df["Age"] = df["Age"].fillna(average_age)

    # Convert specified columns to string type:
    columns_to_convert = ["Pclass", "SibSp", "Parch"]
    df[columns_to_convert] = df[columns_to_convert].astype(str)

    return df


def split_train_test(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) :

    """
    Divides dataframe  into training and test sets.

    Parameters:
      - df: DataFrame, the complete dataset.
      - target_column: str, the name of the column containing the target variable.
      - test_size: float, the size of the test set (default is 0.2).
      - random_state: int, random seed for reproducibility (default is 42).

    Returns:
      - X_train: DataFrame, training features.
      - X_test: DataFrame, test features.
      - y_train: Series, training target variable.
      - y_test: Series, test target variable.
    """

    # Select features and target variable
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def preprocess_generator(X_train):
    # Ejemplo de variables numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Crear los transformadores para las variables numéricas y categóricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Crear el ColumnTransformer para aplicar las transformaciones en un pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit del preprocesador en los datos de entrenamiento
    preprocessor.fit(X_train)

    return preprocessor


def preprocess_applier(preprocessor, X_data):
    # Aplicar el preprocesamiento a los datos
    X_data_processed = preprocessor.transform(X_data)

    # Obtener los nombres de las columnas después del preprocesamiento
    numeric_feature_names = preprocessor.transformers_[0][-1]
    categorical_feature_names = preprocessor.transformers_[1][-1]

    # Obtener las categorías únicas de las variables categóricas
    unique_categories = preprocessor.named_transformers_['cat']['onehot'].categories_

    # Crear los nombres de las columnas después del OneHotEncoding
    encoded_categorical_feature_names = []
    for i, categories in enumerate(unique_categories):
        for category in categories:
            encoded_categorical_feature_names.append(f'{categorical_feature_names[i]}_{category}')

    # Convertir la matriz dispersa a un DataFrame de Pandas
    transformed_df = pd.DataFrame(X_data_processed.toarray(),
                                  columns=numeric_feature_names + encoded_categorical_feature_names)

    return transformed_df