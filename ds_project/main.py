# -*- coding: utf-8 -*-
from loguru import logger

from ds_project.constants import LGBMClassifierParams
from ds_project.data.data import TitanicTrainingData, TitanicTestingData
from ds_project.metrics.metrics import evaluate_model, calculate_roc_auc
from ds_project.models.lgbm import train_lightgbm_model
from ds_project.preprocessing.preprocessing import preprocess_data, split_train_test, preprocess_generator, \
    preprocess_applier



if __name__ == "__main__":
    logger.info("Read Data")
    path = '../data/input/'
    titanic_train = TitanicTrainingData.from_file(path + 'train.csv')

    logger.info("Preprocessing Data")
    titanic_train = preprocess_data(titanic_train.df)

    logger.info("Train Model")
    logger.info("1. Split Dataset")
    X_train, X_test, y_train, y_test = split_train_test(titanic_train, 'Survived')

    logger.info("2. Preprocessing Training Dataset")
    preprocessor = preprocess_generator(X_train)
    X_train = preprocess_applier(preprocessor, X_train)
    X_test = preprocess_applier(preprocessor, X_test)

    logger.info("3. Training Model: LGBM")
    model = train_lightgbm_model(X_train, y_train, X_test, y_test, LGBMClassifierParams.params)

    logger.info("4. Evaluate Model")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluation_metrics = evaluate_model(y_pred, y_test)
    print("metrics:",evaluation_metrics)
    auc_dict = calculate_roc_auc(y_prob, y_test, )
    print("AUC:",auc_dict['auc'])


    logger.info("Predictions")
    titanic_test = TitanicTestingData.from_file(path + 'test.csv')
    titanic_test = preprocess_data(titanic_test.df)
    X_testing = preprocess_applier(preprocessor, titanic_test)
    predictions = model.predict(X_testing)
    titanic_test['Survived'] = predictions
    result = titanic_test[['Survived']].copy()






