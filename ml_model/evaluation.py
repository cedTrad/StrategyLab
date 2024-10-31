from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, log_loss
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt   
import os
import numpy as np


def plot_learning_curve(estimator, X, y, cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'):

  train_sizes, train_scores, test_scores = learning_curve(
      estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring
  )

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  plt.figure(figsize=(10, 6))
  plt.title(f"Learning Curve ({estimator.__class__.__name__})")
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  plt.ylim(0.0, 1.1)
  plt.grid()

  # Tracé de la courbe d'entraînement
  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std, alpha=0.1, color="r")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

  # Tracé de la courbe de validation
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
  plt.legend(loc="best")
  plt.show()



def plot_validation_curve(param_name, param_range, train_scores, test_scores):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 4))
    plt.title(f"Validation Curve with RandomForestClassifier\nParameter: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()
    
    
    
    
class Evaluation:
    
    def __init__(self, model, X_test, X_train, y_test, y_train):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
        self.process()
    
    def process(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_score = self.model.predict_proba(self.X_test)[:, 1]
        self.y_score_train = self.model.predict_proba(self.X_train)[:, 1]
            
    
    def get_metric(self):
        y_test = self.y_test ; y_pred = self.y_pred
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logloss = log_loss(self.y_test, self.y_score)
        
        print(cm)
        print(f"Accuracy : {self.accuracy:.2f}")
        print(f"Taux d'erreur : {1 - self.accuracy:.2f}")
        print(f"Précision: {self.precision:.2f}")
        print(f"Rappel: {self.recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(report)
        print(f"Log Loss: {logloss:.2f}")
        
    
    def auc_roc(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_score)
        auc = roc_auc_score(self.y_test, self.y_score)
        precision, recall, th_pr = precision_recall_curve(self.y_test, self.y_score)
        
        #   --------------------------------
        fpr_train, tpr_train, thresholds_train = roc_curve(self.y_train, self.y_score_train)
        auc_train = roc_auc_score(self.y_train, self.y_score_train)
        precision_train, recall_train, th_pr_train = precision_recall_curve(self.y_train, self.y_score_train)
        
        #   --------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # Courbe ROC
        ax1.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        ax1.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {auc_train:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax1.set_xlabel('Taux de faux positifs (FPR)')
        ax1.set_ylabel('Taux de vrais positifs (TPR)')
        ax1.set_title('Courbe ROC')
        ax1.legend(loc='lower right')
        
        # Courbe de Précision-Rappel
        ax2.plot(recall, precision, marker='.')
        ax2.plot(recall_train, precision_train)
        ax2.set_xlabel('Rappel')
        ax2.set_ylabel('Précision')
        ax2.set_title('Courbe de Précision-Rappel')

        plt.show()








