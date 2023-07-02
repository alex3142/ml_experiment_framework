import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBClassifier, XGBRegressor


data = load_diabetes()

features_df = pd.DataFrame(data=data.data, columns=data.feature_names)
labels_df = pd.DataFrame(data=data.target, columns=["label"])
#labels_df["binary_label"] = (labels_df["multi_label"] == 0).astype(int)

kf = KFold()

random_seed = 3142
fold_count = 0
for train_index, val_index in kf.split(features_df):
    X_train, y_train = features_df.iloc[train_index], labels_df["label"].iloc[train_index]
    X_val, y_val = features_df.iloc[val_index], labels_df["label"].iloc[val_index]

    reg = XGBRegressor(early_stopping_rounds=20, eval_metric="rmse", max_depth=4, learning_rate=0.7)

    fold_count += 1

    print(f"fold = {fold_count}")

    X_train, X_early_stop, y_train, y_early_stop = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_seed*fold_count
    )

    reg.fit(X_train, y_train, eval_set=[(X_early_stop, y_early_stop)])

    best_iteration = reg.best_iteration

    print(f"best_ iteration = {best_iteration}")

    reg = XGBRegressor(n_estimators=best_iteration, max_depth=4, learning_rate=0.1)
    reg.fit(X_train, y_train)

    print(f"rmse = {mean_squared_error(y_true=y_val,y_pred=reg.predict(X_val)) ** 0.5}")
    print("       ")