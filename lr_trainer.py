import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from pdb import set_trace as st

class LRTrainer():
    def preprocess_train(self, raw_X, raw_y):
        print('Preprocessing X train...')
        X = np.sum(np.array(raw_X), axis = 2)
        if self.interaction:
            interaction = PolynomialFeatures(interaction_only=True)
            X = interaction.fit_transform(X)
        y = np.argmax(np.array(raw_y), axis = 1)
        return X,y 

    def preprocess_test(self, raw_X_test):
        X_test = np.sum(np.array(raw_X_test), axis = 2)
        if self.interaction:
            interaction = PolynomialFeatures(interaction_only=True)
            X_test = interaction.fit_transform(X_test)
        return X_test

    def train(self, raw_X, raw_y, interaction):
        self.interaction = interaction
        X, y = self.preprocess_train(raw_X, raw_y)

        print('Grid search cross validation...')
        parameters = {'C':[2**x for x in range(-15,5)]}
        lr = LogisticRegression(multi_class='multinomial', penalty='l2', 
            tol=0.000001, max_iter=1000, solver='lbfgs')
        cv_clf = GridSearchCV(lr, param_grid = parameters)
        cv_clf.fit(X, y)
        best_C = cv_clf.best_params_['C']
        
        print('Running logistic regression...')
        self.clf = LogisticRegression(multi_class='multinomial', C=best_C,
            penalty='l2', tol=0.000001, max_iter=1000, solver='lbfgs')
        self.clf.fit(X, y)

    def predict(self, raw_X_test):
        X_test = self.preprocess_test(raw_X_test)
        return self.clf.predict_proba(X_test)