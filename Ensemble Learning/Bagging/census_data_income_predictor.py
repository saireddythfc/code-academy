import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

col_names = ['age', 'workclass', 'fnlwgt','education', 'education-num', 
'marital-status', 'occupation', 'relationship', 'race', 'sex',
'capital-gain','capital-loss', 'hours-per-week','native-country', 'income']
df = pd.read_csv('adult.data', header=None, names = col_names)

#Distribution of income
print(df.income.value_counts(normalize = True))

#Clean columns by stripping extra whitespace for columns of type "object"
cols = df.select_dtypes(include = object)
for col in cols:
  df[col] = df[col].str.strip()

#Create feature dataframe X with feature columns and dummy variables for categorical features
feature_cols = ['age',
       'capital-gain', 'capital-loss', 'hours-per-week', 'sex','race']
X = pd.get_dummies(df[feature_cols], drop_first = True)
#Create output variable y which is binary, 0 when income is less than 50k, 1 when it is greather than 50k
y = pd.Series([0 if x == "<=50K" else 1 for x in df.income])

#Split data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Instantiate random forest classifier, fit and score with default parameters
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f'Accuracy score for default random forest: {round(rf.score(X_test, y_test)*100,3)}%')

#Tune the hyperparameter max_depth over a range from 1-25, save scores for test and train set
np.random.seed(0)
accuracy_train=[]
accuracy_test = []
depths = []
for depth in range(1, 26):
  depths.append(depth)
  rfm = RandomForestClassifier(max_depth=depth)
  rfm.fit(X_train, y_train)
  accuracy_train.append(accuracy_score(y_train, rfm.predict(X_train)))
  accuracy_test.append(accuracy_score(y_test, rfm.predict(X_test)))
  

#Find the best accuracy and at what depth that occurs
best_accuracy_test = max(accuracy_test)
best_depth = accuracy_test.index(best_accuracy_test) + 1
print(f'The highest accuracy on the test is achieved when depth: {best_depth}')
print(f'The highest accuracy on the test set is: {round(best_accuracy_test*100,3)}%')

#Plot the accuracy scores for the test and train set over the range of depth values 
plt.plot(depths, accuracy_test,'bo--',depths, accuracy_train,'r*:')
plt.legend(['test accuracy', 'train accuracy'])
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.show()


#Save the best random forest model and save the feature importances in a dataframe
best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(X_train, y_train)
feature_imp_df = pd.DataFrame(zip(X_train.columns, best_rf.feature_importances_),  columns=['feature', 'importance'])
print('Top 5 random forest features:')
print(feature_imp_df.sort_values('importance', ascending=False).iloc[0:5])


#Create two new features, based on education and native country
df['education_bin'] = pd.cut(df['education-num'], [0,9,13,16], labels=['HS or less', 'College to Bachelors', 'Masters or more'])

feature_cols = ['age',
        'capital-gain', 'capital-loss', 'hours-per-week', 'sex', 'race','education_bin']
X = pd.get_dummies(df[feature_cols], drop_first=True)
#Use these two new additional features and recreate X and test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Find the best max depth now with the additional two features
np.random.seed(0)
accuracy_train=[]
accuracy_test = []
depths = []
for depth in range(1, 26):
  depths.append(depth)
  rfm = RandomForestClassifier(max_depth=depth)
  rfm.fit(X_train, y_train)
  accuracy_train.append(accuracy_score(y_train, rfm.predict(X_train)))
  accuracy_test.append(accuracy_score(y_test, rfm.predict(X_test)))
  

#Find the best accuracy and at what depth that occurs
best_accuracy_test = max(accuracy_test)
best_depth = accuracy_test.index(best_accuracy_test) + 1
print(f'The highest accuracy on the test is achieved when depth: {best_depth}')
print(f'The highest accuracy on the test set is: {round(best_accuracy_test*100,3)}%')

#Plot the accuracy scores for the test and train set over the range of depth values 
plt.plot(depths, accuracy_test,'bo--',depths, accuracy_train,'r*:')
plt.legend(['test accuracy', 'train accuracy'])
plt.xlabel('max depth')
plt.ylabel('accuracy')
plt.show()




#Save the best model and print the two features with the new feature set
best_rf = RandomForestClassifier(max_depth=best_depth)
best_rf.fit(X_train, y_train)
feature_imp_df = pd.DataFrame(zip(X_train.columns, best_rf.feature_importances_),  columns=['feature', 'importance'])
print('Top 5 random forest features:')
print(feature_imp_df.sort_values('importance', ascending=False).iloc[0:5])



"""

There are a few different ways to extend this project:

Are there other features that may lead to an even better performace? Consider creating new ones or adding additional features not part of the original feature list.
Consider tuning hyperparameters based on a different evaluation metric – our classes are fairly imbalanced, AUC of F1 may lead to a different result
Tune more parameters of the model. You can find a description of all the parameters you can tune in the Random Forest Classifier documentation(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). For example, see what happens if you tune max_features or n_estimators.

"""