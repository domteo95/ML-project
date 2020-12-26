from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, \
    plot_confusion_matrix, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('crime.csv')

df['day'] = df['day'].apply(lambda x: 0 if x == 'WEEKDAY' else 1)
df = df.drop(['Arrest', 'Domestic'], axis=1)


# split training and testing dataset
labels = df.iloc[:, 2:].columns.values
features = df.iloc[:, 2:].values
target = df['Primary Type'].values

X_train, X_test, y_train, y_test = train_test_split(features, target, \
    test_size=0.90, random_state=158) # the data is too large so we train with 10% data


# standardize continuous variables
x_train_df = pd.DataFrame(X_train, columns=labels)
sc = StandardScaler()
sd_x_trian_df = pd.DataFrame(sc.fit_transform(x_train_df.iloc[:, :4]), columns=labels[:4])
sub_train_df = x_train_df.iloc[:, 4:]
x_train_df = pd.concat([sd_x_trian_df, sub_train_df], axis=1)
X_train = x_train_df.values

x_test_df = pd.DataFrame(X_test, columns=labels)
sd_x_test_df = pd.DataFrame(sc.transform(x_test_df.iloc[:, :4]), columns=labels[:4])
sub_test_df = x_test_df.iloc[:, 4:]
x_test_df = pd.concat([sd_x_test_df, sub_test_df], axis=1)
X_test = x_test_df.values


# gridsearch
pipeline = Pipeline(steps=[('imputation', SimpleImputer(strategy='median', copy=False)), \
                           ('lgr', LogisticRegression(max_iter=5000, n_jobs=4, , class_weight={0:0.86, 1:1.2}))])
scores = {'Accuracy': make_scorer(accuracy_score), \
          'Precision': make_scorer(precision_score, pos_label="VIOLENT CRIME"),\
          'Recall': make_scorer(recall_score, pos_label="VIOLENT CRIME")}
params = {'lgr__C':[0.5, 0.75, 1, 1.25, 1.5], 'lgr__solver':['lbfgs', 'liblinear', 'sag', 'saga']}
tscv = TimeSeriesSplit(n_splits=5)

grid = GridSearchCV(pipeline, params, scoring=scores, cv=tscv, refit='Accuracy')
grid.fit(X_train, y_train)


# see scores
parameters = pd.DataFrame(grid.cv_results_['params'])
accuracy = pd.DataFrame(grid.cv_results_['mean_test_Accuracy'])
precision = pd.DataFrame(grid.cv_results_['mean_test_Precision'])
recall = pd.DataFrame(grid.cv_results_['mean_test_Recall'])
score_df = pd.concat([parameters, accuracy, precision, recall], axis=1)
score_df.clumns = ['C', 'solver', 'Accuracy', 'Precision', 'Recall']


# best model
grid.best_estimator_


# train best model
imp = SimpleImputer(strategy='median', copy=False)
X_train = imp.fit_transform(X_train)

model = LogisticRegression( C=1, solver='lbfgs', max_iter=5000, n_jobs=4, \
    class_weight= {0:0.86, 1:1.2})
model.fit(X_train, y_train)


# test
X_test = imp.transform(X_test)
y_pred = model.predict(X_test)


# evaluation
plot_confusion_matrix(model, X_test, y_test)
plt.show()

print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))