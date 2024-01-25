"""
The beans data set comes from the UCI Machine Learning Repository. The original data set contains data for seven different kinds of beans, but for the sake of simplicity you’ll only look at two classes: barbunya beans and cali beans. The task is to classify beans based on their size and shape. There are 16 independent variables.

Use scikit-learn to create an LDA model called lda. Use the fit_transform() method to fit lda to X and y and create a 1-dimensional subspace called X_new.

Fit the logistic regression model to the new subspace X_new and y. Then create a variable called lr_acc and set it to the accuracy of the logistic regression model. You can use the score() method to find the accuracy. Then print lr_acc.

"""

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data
beans = pd.read_csv('beans.csv')
X = beans.drop('Class', axis=1)
y = beans['Class']

# Create an LDA model
lda = LinearDiscriminantAnalysis(n_components = 1)

# Fit lda to X and y and create a subspace X_new
X_new = lda.fit_transform(X, y)

# Create a logistic regression model
lr = LogisticRegression()

# Fit lr to X_new and y
lr.fit(X_new, y)

# Model accuracy
lr_acc = lr.score(X_new, y)
print(lr_acc)

