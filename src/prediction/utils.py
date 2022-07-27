from typing import List

import pandas as pd

from src.constants import FeaturesConstants


def predict_for_exist(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: List[str] = FeaturesConstants.target,
) -> pd.DataFrame:
    train_and_tests_ctr_ = list(set(train['page_id']) & set(test['page_id']))
    DUPLI_X_TRAIN = (
        train[train['page_id'].isin(train_and_tests_ctr_)].reset_index().copy()
    )
    DUPLI_Y_TRAIN = (
        train[train['page_id'].isin(train_and_tests_ctr_)]
        .reset_index()
        .copy()[target]
    )

    DUPLI_X_TEST = test[test['page_id'].isin(train_and_tests_ctr_)].copy()
    prediction = {}
    for i in train_and_tests_ctr_:
        df1 = DUPLI_X_TRAIN[DUPLI_X_TRAIN['page_id'] == i].fillna(-1)
        df2 = DUPLI_X_TEST[DUPLI_X_TEST['page_id'] == i].fillna(-1)

        for j, row in df2.iterrows():
            prediction[j] = {}
            prediction[j]['document_id'] = row['document_id']
            #         prediction[j]['ctr'] = row['ctr']
            if i == '62544e1c9a7947e9895bcb1d':
                db = df1[target]
            else:
                db = df1[df1['ctr'] == row['ctr']][target]
            for i in target:
                try:
                    prediction[j][i] = db[i].values[0]
                except Exception as ex:
                    print(row['document_id'], row['page_id'])
    pred_intersection = pd.DataFrame(prediction).T
    return pred_intersection
