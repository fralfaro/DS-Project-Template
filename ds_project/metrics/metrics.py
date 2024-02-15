from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, roc_curve

def evaluate_model(y_pred, y_test):
    """
    Evaluates a classification model using accuracy, precision, recall, and F1-score.

    Parameters:
        y_pred: numpy array or pandas Series, predicted labels.
        y_test: numpy array or pandas Series, true labels.

    Returns:
        dict: dictionary containing evaluation metrics.
    """

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Remove unnecessary time measurement
    # (it's not directly relevant to the model evaluation)

    # Return evaluation metrics
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

def calculate_roc_auc(y_prob, y_test, ):
  """
  Calculates ROC AUC and returns associated metrics.

  Parameters:
      model: Trained model object with predict_proba method.
      X_test: pandas DataFrame, test features.
      y_test: pandas Series, test target variable.
      pos_label: int, label considered positive (default: 1).

  Returns:
      dict: dictionary containing ROC AUC and curve data.
  """

  # Calculate ROC curve and AUC
  fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
  auc = round(roc_auc_score(y_test, y_prob), 3)

  # Return combined metrics
  return {"fpr": fpr, "tpr": tpr, "auc": auc}

