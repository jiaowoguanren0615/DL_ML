import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from catboost import CatBoostClassifier


def add_original_data(original, train, target, label=1):
    target_dict = {k: v for k, v in zip(original[target].unique(), list(range(len(original[target].unique()))))}
    original[target] = original[target].map(target_dict)
    if label == 2:  # all labels
        train = pd.concat([train, original], ignore_index=True)
    else:  # label 0 or 1
        train = pd.concat([train, original[original[target] == label]], ignore_index=True)
    return train.drop_duplicates()


def categorical_encode(train, test, cat_cols, type='onehot'):
    if type == "onehot":
        train = pd.get_dummies(train, prefix=cat_cols, drop_first=True)
        test = pd.get_dummies(test, prefix=cat_cols, drop_first=True)
    else:
        le = LabelEncoder()
        for col in cat_cols:
            train[col] = le.fit_transform(train[[col]])
            test[col] = le.transform(test[[col]])
    return train, test


def feature_scale(train, test, cols, scalar_idx=0):
    scaler_list = [MinMaxScaler(), StandardScaler()]
    scaler = scaler_list[scalar_idx]
    for col in cols:
        train[col] = scaler.fit_transform(train[[col]])
        test[col] = scaler.transform(test[[col]])
    return train, test


def validate(X, y, X_test, model):
    score_list, oof_pred = [], []
    kf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        print(f"Fold {fold} validating...")
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        oof_pred.append(model.predict_proba(X_test)[:, 1])
        score = roc_auc_score(y_val, y_pred)
        score_list.append(score)
        print(f"AUC: {score:.4f}")
    print(f"Average AUC: {sum(score_list) / len(score_list):.4f}")
    print(f"{score_list}")
    return sum(oof_pred) / kf.get_n_splits()


def feature_engineering(train, test, train_org, target):
    train = add_original_data(train_org.copy(), train, 1)
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'Over18',
                'OverTime']
    train, test = categorical_encode(train, test, cat_cols, type='onehot')

    drop_cols = [target]
    features = [col for col in train.columns if col not in drop_cols]

    X = train[features]
    X_test = test[features]
    y = train[target]

    X, X_test = feature_scale(X, X_test, X.columns)
    return X, y, X_test

