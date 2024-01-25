"""
We can also use LDA itself as a classifier. In this case, scikit-learn will project the data onto a subspace and then find a decision boundary that is perpendicular to that subspace. Put simply, it will do LDA and use the result to find a linear decision boundary.

Since we’ve already fit lda to X and y, we can simply look at the accuracy of the model by using the score() method.
"""

# Import libraries
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data
beans = pd.read_csv('beans.csv')
X = beans.drop('Class', axis=1)
y = beans['Class']

# Create LDA model
lda = LinearDiscriminantAnalysis(n_components=1)

# Fit the data and create a subspace X_new
lda.fit(X, y)

# Print LDA classifier accuracy
lda_acc = lda.score(X, y)
print(lda_acc)
