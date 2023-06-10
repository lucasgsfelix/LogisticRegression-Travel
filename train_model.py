
import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def split_data_k_folds(X, y, k, train_equalizer=True):

    skf = KFold(n_splits=k)
    folds = []
    
    # Calcula as proporções das classes
    class_counts = np.unique(y, return_counts=True)[1]
    class_proportions = class_counts / len(y)
    
    for train_index, test_index in skf.split(X, y):
        
        
        if train_equalizer:
    
            X_train = X.loc[train_index]
            X_train['trip_type'] = y.loc[train_index]

            class_zero = X_train[X_train['trip_type'] == 0]
            class_one = X_train[X_train['trip_type'] == 1]

            X_train = class_zero.sample(len(class_one)).append(class_one)

            fold_train_X = X_train.drop('trip_type', axis=1)
            fold_train_y = X_train['trip_type']

            fold_test_X = X.loc[test_index]
            fold_test_y = y.loc[test_index]

            # Verifica as proporções das classes no conjunto de treinamento atual
            fold_train_class_counts = np.unique(fold_train_y, return_counts=True)[1]
            fold_train_class_proportions = fold_train_class_counts / len(fold_train_y)
    
        else:
            
            fold_train_X = X.loc[train_index]
            fold_train_y = y.loc[train_index]
            
            fold_test_X = X.loc[test_index]
            fold_test_y = y.loc[test_index]

        folds.append((fold_train_X, fold_train_y, fold_test_X, fold_test_y))
    
    return folds


def train_model(folds, cut_off):


    micro, macro = [], []


    test_predictions, y_test_values = [], []

    for i, fold in enumerate(folds):
        
        x_train, y_train, x_test, y_test = fold
        
        clf = LogisticRegression(penalty='l2', max_iter=1000, class_weight='balanced',
                                solver='newton-cholesky').fit(x_train, y_train)
        

        predictions = clf.predict_proba(x_test)
        
        
        predictions = [1 if prediction[1] >= 0.5 else 0 for prediction in predictions]
        
        #print("Micro ", f1_score(predictions, y_test, average='micro'))
        #print("Macro ", f1_score(predictions, y_test, average='macro'))

        micro.append(f1_score(predictions, y_test, average='micro'))
        macro.append(f1_score(predictions, y_test, average='macro'))
        
        test_predictions.extend(predictions)
        
        y_test_values.extend(y_test)
        

    micro_df = pd.DataFrame(micro).T.rename(columns={fold: str(fold) + '_micro' for fold in range(0, 5)})

    macro_df = pd.DataFrame(macro).T.rename(columns={fold: str(fold) + '_macro' for fold in range(0, 5)})

    cm = confusion_matrix(y_test_values, test_predictions, labels=clf.classes_)

    metrics_df = micro_df.join(macro_df)

    metrics_df['True Positive'] = cm[0][0]
    metrics_df['False Negative'] = cm[0][1]
    metrics_df['False Positive '] = cm[1][0]
    metrics_df['True Negative'] = cm[1][1]

    metrics_df['cutoff'] = cut_off

    return metrics_df


if __name__ == '__main__':


    cut_off = sys.argv[1]

    df = pd.read_table("user_trips_table_" + cut_off + ".csv", sep=';')

    columns_to_drop = ['trip_type', 'user_id','trip_id', 'date']

    folds = split_data_k_folds(df.drop(columns_to_drop, axis=1), df['trip_type'], 5, False)

    metrics_df = train_model(folds, cut_off)



    if not "model_metrics.csv" in os.listdir("Datasets"):

        metrics_df.to_csv("Datasets/model_metrics.csv", sep=';', index=False, header=True, mode='w')

    else:

        metrics_df.to_csv("Datasets/model_metrics.csv", sep=';', index=False, header=False, mode='a')
