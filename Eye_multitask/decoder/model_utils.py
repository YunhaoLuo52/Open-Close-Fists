import numpy as np
import pandas as pd
import pickle
from typing import OrderedDict
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score,confusion_matrix
from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.base import BaseEstimator, TransformerMixin


class AverageChannels(BaseEstimator, TransformerMixin):
    """
    Average the channels 
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_avg = X.mean(axis=1, keepdims=True)
        return X_avg


class model:
    def __init__(self):
        # Note that the pipeline averages the TP9 and TP10 channels 
        # if using LR or LDA
        self.clfs = OrderedDict()
        self.clfs['Vect + SVM'] = make_pipeline(
            AverageChannels(),
            Vectorizer(),
            StandardScaler(),
            svm.SVC(probability=True)
        )
        self.clfs['Vect + PCA + SVM'] = make_pipeline(
            AverageChannels(),
            Vectorizer(),
            StandardScaler(),
            PCA(n_components=10),
            svm.SVC(probability=True)
        )
        self.clfs['Vect + LR'] = make_pipeline(
            AverageChannels(),
            Vectorizer(),
            StandardScaler(),
            LogisticRegression()
        )
        # self.clfs['Vect + RegLDA'] = make_pipeline(
        #     AverageChannels(),
        #     Vectorizer(),
        #     LDA(shrinkage='auto', solver='eigen')
        # )
        self.clfs['ERPCov + TS'] = make_pipeline(
            ERPCovariances(),
            TangentSpace(),
            LogisticRegression()
        )
        # self.clfs['ERPCov + MDM'] = make_pipeline(
        #     ERPCovariances(),
        #     MDM()
        # )
        
        self.cv_results_ = None
        self.best_model_name_ = None
        self.best_model_ = None
        self.best_score_ = None
    
    def model_selection(self, X, y, scoring='roc_auc', n_splits=10,
                        test_size=0.25, random_state=42):
        """
        cross validate and pick best model
        """
        cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )

        auc_scores = []
        methods = []

        for method_name, clf in self.clfs.items():
            scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            auc_scores.extend(scores)
            methods.extend([method_name] * len(scores))

        self.cv_results_ = pd.DataFrame({
            'Method': methods,
            'Score': auc_scores
        })

        mean_scores = self.cv_results_.groupby('Method')['Score'].mean()
        self.best_model_name_ = mean_scores.idxmax()  
        self.best_score_ = mean_scores.max()           
        self.best_model_ = self.clfs[self.best_model_name_]  

        print(f"Best model after CV: {self.best_model_name_} with mean {scoring}={self.best_score_:.4f}")

        return self.best_model_name_
    
   
    def train(self, X_train:np.array, y_train:np.array):
        if self.best_model_ is None:
            raise RuntimeError("No best model selected yet. Call model_selection() or set_best_model() first.")
        self.best_model_.fit(X_train, y_train)
        print(f"Trained best model: {self.best_model_name_}")

    def set_best_model(self, method_name):
        """
        set best model manually
        """
        if method_name not in self.clfs:
            raise ValueError(f"Method {method_name} not in candidate classifiers!")
        self.best_model_name_ = method_name
        self.best_model_ = self.clfs[method_name]
        self.best_score_ = None  

    def predict(self, X_test:np.array):
        if self.best_model_ is None:
            raise RuntimeError("No best model available.")
        return self.best_model_.predict(X_test)
    
    def evaluate(self, X_test: np.array, y_test: np.array, metric: str='roc_auc'):
        """
        - metric='roc_auc'
        - metric='accuracy'
        """
        if self.best_model_ is None:
            raise RuntimeError("No best model available.")
        

        y_pred = self.best_model_.predict(X_test)
        if metric == 'roc_auc':
            if hasattr(self.best_model_, 'predict_proba'):
                y_proba = self.best_model_.predict_proba(X_test)[:, 1] 
                score = roc_auc_score(y_test, y_proba)
                return score, y_proba
            else:
                raise ValueError("This model doesn't support predict_proba, can't compute ROC AUC.")
        elif metric == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif metric == 'confusion_matrix':
            score=confusion_matrix(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if metric == 'confusion_matrix':
            print("Confusion matrix on the test set is: ")
            print(score)
        else:
            print(f"Evaluation on test set ({metric}): {score:.4f}")
        return score
    
    def save_model(self, fp:str):
        with open(fp, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {fp}")
    
    @classmethod
    def load_model(cls, fp):
        with open(fp, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"Model loaded from {fp}")
        return loaded_model