import numpy as np
import pandas as pd
#import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_quality.csv')
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
ss_fit = StandardScaler().fit(features)
X = ss_fit.transform(features)

## 2. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 99)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg = LogisticRegression(penalty = 'none')
clf_no_reg.fit(X_train, y_train)

## 4. Plot the coefficients
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()
coef.plot(kind = 'bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()

## 5. Training and test performance
from sklearn.metrics import f1_score

y_preds_train = clf_no_reg.predict(X_train)
f1_train = f1_score(y_preds_train, y_train)

y_preds_test = clf_no_reg.predict(X_test)
f1_test = f1_score(y_preds_test, y_test)

print('Training Score', f1_train)
print('Testing Score', f1_test)


## 6. Default Implementation (L2-regularized!)
clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)

## 7. Ridge Scores
y_preds_train = clf_default.predict(X_train)
f1_train = f1_score(y_preds_train, y_train)

y_preds_test = clf_default.predict(X_test)
f1_test = f1_score(y_preds_test, y_test)

print('Ridge training Score', f1_train)
print('Ridge testing Score', f1_test)


## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
for x in C_array:
    clf = LogisticRegression(C = x )
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    training_array.append(f1_score(y_train, y_pred_train))
    test_array.append(f1_score(y_test, y_pred_test))


## 9. Plot training and test scores as a function of C
plt.plot(C_array, training_array)
plt.plot(C_array, test_array)
plt.xscale('log')
plt.show()
plt.clf()


## 10. Making a parameter grid for GridSearchCV
C_array = np.logspace(-4, -2, 100)
tuning_C = {'C': C_array}


## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
gsm = GridSearchCV(clf, param_grid = tuning_C, scoring = 'f1', cv = 5)
gsm.fit(X_train, y_train)


## 12. Optimal C value and the score corresponding to it
print(gsm.best_params_, gsm.best_score_)

## 13. Validating the "best classifier"
best_C = gsm.best_params_['C']
clf_best_ridge = LogisticRegression(C = best_C)
clf_best_ridge.fit(X_train, y_train)
y_preds = clf_best_ridge.predict(X_test)
print("Best model f1-score: ", f1_score(y_preds, y_test))

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
C_array = np.logspace(-2,2,100)
clf_l1 = LogisticRegressionCV(Cs=C_array, cv = 5, penalty = 'l1', scoring = 'f1', solver = 'liblinear')
clf_l1.fit(X,y)


## 15. Optimal C value and corresponding coefficients
print('Best C value', clf_l1.C_)
print('Best fit coefficients', clf_l1.coef_)


## 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients, predictors).sort_values()
coef.plot(kind = 'bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()
