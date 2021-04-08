"""
    Compute the cavs by training a model of choice against a concept vector and labels (X, y).
"""

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def split_the_data(X, y):
    """
    Splits the data into train and test sets to compute the scores against.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)

def fit_the_model(X, y, model_type="LR"):
    """
    Fits the model on the pair of X and y.
    
    Arguments
        X (array): the 2D array of samples
        y (array): the 1D array of labels
        model_type (str): the type of model to train
            LR = LogisticRegression
            END = ElasticNet with Default parameters
            ENM = ElasticNet with modified parameters

    Returns
        lm (the linear model): the fit sklean linear model.
    """
    if model_type == "LR":
        lm = linear_model.LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=10000)
    elif model_type == "END":
        lm = linear_model.ElasticNet(random_state=0)
    elif model_type == "ENM":
        lm = linear_model.ElasticNet(alpha=0.002)
    elif model_type == "SGDC":
        lm = linear_model.SGDClassifier(loss='log') # iterative logistic regression.
        
    lm.fit(X, y)
    return lm

def compute_accuracy(lm, X_test, y_test):
    """
    Given a test set, compute the accuracy of the model as a sanity check.

    Arguments
        X (array): the 2D array of samples
        y (array): the 1D array of labels
    
    Returns
        score (int): the accuracy score.
    """
    y_pred = lm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

def get_cav(lm):
    """
    Get the CAV of the fit linear model.

    Arguments
        lm (the linear model): the fit sklean linear model.
    
    Returns
        cavs (ndarray): the computed coef_ (cav) of the model.
    """
    cavs = lm.coef_.ravel()
    # since this is binary classification, by default the concept is assigned to label 0, 
    # so multiply by -1 to assign it to label 1 (concept).
    return -1*cavs

def run(X, y, model_type):
    """
    Run the above fts in a sequence and return the computed CAVs
    
    Arguments
        X (array): the 2D array of samples
        y (array): the 1D array of labels
        model_type (str): the type of model to train

    Returns
        cavs (ndarray): the computed coef_ (cav) of the model.
        accuracy (float): the accuracy score of the model.
    """
    X_train, X_test, y_train, y_test = split_the_data(X, y)
    lm = fit_the_model(X, y, model_type)
    test_accuracy = compute_accuracy(lm, X_test, y_test)
    cav = get_cav(lm)

    return (cav, test_accuracy)