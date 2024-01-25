import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold 

rand_state = 42

#using only ph, conductivity, and turbidity in exercise.
water_potability = pd.read_csv('water_potability')
print(water_potability.columns, water_potability.shape)

X = water_potability.drop(['Potability'], axis=1)
y = water_potability['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rand_state)

level_0_estimators = dict()
level_0_estimators["logreg"] = LogisticRegression(random_state=rand_state)
level_0_estimators["forest"] = RandomForestClassifier(random_state=rand_state)

level_0_columns = [f"{name}_prediction" for name in level_0_estimators.keys()]

level_1_estimator = RandomForestClassifier(random_state=rand_state)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=rand_state)
stacking_clf = StackingClassifier(estimators=list(level_0_estimators.items()), 
                                    final_estimator=level_1_estimator, 
                                    passthrough=True, cv=kfold, stack_method="predict_proba")

df = pd.DataFrame(stacking_clf.fit_transform(X_train, y_train), columns=level_0_columns + list(X_train.columns))

y_val_pred = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_val_pred)

vanilla_logistic_regression = LogisticRegression(random_state=rand_state).fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, vanilla_logistic_regression.predict(X_test))
                                   
vanilla_decision_tree = RandomForestClassifier(random_state=rand_state).fit(X_train, y_train)
dt_accuracy =  accuracy_score(y_test, vanilla_decision_tree.predict(X_test))

print(f'Stacking accuracy: {stacking_accuracy:.4f}')
print(f'Logistic Regression accuracy: {lr_accuracy:.4f}')
print(f'Decision Tree accuracy: {dt_accuracy:.4f}')



