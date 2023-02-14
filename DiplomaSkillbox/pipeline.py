import os
import dill
import sklearn
import imblearn
import datetime
import pandas as pd

from sklearn.pipeline import Pipeline as PipelineSK
from imblearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier

def features_adding(data):
    import math

    data = data.copy()
    data.device_screen_resolution = data.device_screen_resolution.apply(lambda x: x.split('x'))
    data['device_screen_resolution_log'] = data.device_screen_resolution.apply(
        lambda x: round(math.log(int(x[0]) * int(x[1])), 2) if (int(x[0]) * int(x[1])) != 0 else 0)
    data['device_screen_resolution_log'] = data['device_screen_resolution_log'].replace(0, data['device_screen_resolution_log'].median())
    data.visit_time = pd.to_datetime(data.visit_time)
    data['hour'] = data.visit_time.dt.hour
    data.visit_date = pd.to_datetime(data.visit_date)
    data['month'] = data.visit_date.dt.month
    data['dayofweek'] = data.visit_date.dt.weekday

    data['n_days_from_start'] = data.visit_date - data.visit_date.min()
    data['n_days_from_start'] = data['n_days_from_start'].astype(str).apply(lambda x: x.split(' ')[0]).astype('int64')

    data['visit_number_log'] = data.visit_number.apply(lambda x: math.log(x))

    data['utm_generic'] = data['utm_campaign'].astype(str) + '_' + data['month'].astype(str)
    return data


def drop_useless_columns(data):
    # Удаляем неинформативные колонки
    columns_to_drop = [
        'visit_date',
        'visit_time',
        'visit_number',
        'utm_campaign',
        'device_screen_resolution',
        'month'
    ]
    return data.drop(columns_to_drop, axis=1)





def main():

    df = pd.read_csv(f'data/sberpodpiska_data_prepared.csv')

    X = df.drop(['event_action_bin'], axis=1)
    y = df['event_action_bin']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    #написание пайплайна
    feature_transform = ColumnTransformer(transformers=[
        ('numerical_transform', StandardScaler(), make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical_transform', TargetEncoder(smoothing=0.5), make_column_selector(dtype_include=object))],
         n_jobs=-1)

    preprocessor = PipelineSK(steps=[
        ('features_add', FunctionTransformer(features_adding)),
        ('drop_columns', FunctionTransformer(drop_useless_columns)),
        ('transformation', feature_transform)
    ])

    models = (


        HistGradientBoostingClassifier(l2_regularization=30, max_iter=600, learning_rate=0.081, random_state=42),
        LogisticRegression(C=1E6, penalty='l2', solver='lbfgs', n_jobs=-1, max_iter=1000, class_weight={0: 0.5, 1: 13},
                           random_state=42),
        RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_split=8, n_jobs=-1, n_estimators=500,
                               class_weight='balanced', random_state=42),
        CatBoostClassifier(depth=10, learning_rate=0.04, iterations=500,
                           devices='0:1', class_weights={0: .5, 1: 13}),
        StackingClassifier(estimators=[

            ('HistGr', HistGradientBoostingClassifier(l2_regularization=30, max_iter=500, learning_rate=0.081,
                                                      random_state=42)),
            ('logreg', LogisticRegression(C=1E6, penalty='l2', solver='lbfgs', n_jobs=-1, max_iter=1000,
                                          class_weight={0: .5, 1: 13}, random_state=42))],
            final_estimator=RandomForestClassifier(max_depth=8, max_features='sqrt', min_samples_split=8, n_jobs=-1,
                                                   n_estimators=500, class_weight={0: .5, 1: 13}, random_state=42),
            n_jobs=-1)

    )

    best_score = .0
    best_pipe = None
    best_pred = []
    for model in models:
        pipe = PipelineSK(steps=[
            ('preprocessor', preprocessor),
            ('sample_classifier_pipeline', Pipeline(steps=[
                ('classifier', model)
            ]))
        ])

        pipe.fit(train_x, train_y)
        proba = pipe.predict_proba(test_x)[:, 1]
        pred = pipe.predict(test_x)
        score = roc_auc_score(test_y, proba)
        print(f'model: {type(model).__name__}, roc_auc: {score:.4f}')
        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_pred = pred

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps.sample_classifier_pipeline.named_steps["classifier"]).__name__}, '
          f'roc_auc_score: {best_score:.4f}\n')
    print('Матрица ошибок:\n\n', confusion_matrix(test_y, best_pred), '\n')
    print('Значения метрик:\n', classification_report(test_y, best_pred))


    with open('models/sberpodpiska.pkl', 'wb') as file:
        dill.dump({
           'model': best_pipe,
           'metadata': {
                'name': 'Sberpodpiska',
                'author': 'Dmitry Troynin',
                'version': 1.0,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps.sample_classifier_pipeline.named_steps["classifier"]).__name__,
                'roc_auc_score': best_score
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

#%%
