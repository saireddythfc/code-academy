# # Classify Raisins with Hyperparameter Tuning Project
# 
# - [View Solution Notebook](./solution.html)
# - [View Project Page](https://www.codecademy.com/projects/practice/mle-hyperparameter-tuning-project)

# ### 1. Explore the Dataset

# In[4]:


# 1. Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

raisins = pd.read_csv("raisins.csv")
raisins.head()


# In[5]:


# 2. Create predictor and target variables, X and y
y = raisins.pop(&#39;Class&#39;)
X = raisins


# In[7]:


# 3. Examine the dataset
n_features = len(X.columns)
n_samples = len(X)
n_samples_c1 = sum(y)


# In[8]:


print(n_features, n_samples, n_samples_c1)


# In[9]:


# 4. Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 19)


# ### 2. Grid Search with Decision Tree Classifier

# In[10]:


# 5. Create a Decision Tree model
tree = DecisionTreeClassifier()


# In[11]:


# 6. Dictionary of parameters for GridSearchCV
params = {&#39;max_depth&#39;: [3, 5, 7], &#39;min_samples_split&#39;: [2, 3, 4]}


# In[15]:


# 7. Create a GridSearchCV model
gsm = GridSearchCV(tree, params)

#Fit the GridSearchCV model to the training data
gsm.fit(X_train, y_train)


# In[16]:


# 8. Print the model and hyperparameters obtained by GridSearchCV
best_score = gsm.best_score_
test_score = gsm.score(X_test, y_test)

# Print best score
print(best_score)

# Print the accuracy of the final model on the test data
print(test_score)


# In[21]:


# 9. Print a table summarizing the results of GridSearchCV
hyper_params = gsm.cv_results_[&#39;params&#39;]
hyper_parameters = pd.DataFrame.from_dict(hyper_params)

scores = gsm.cv_results_[&#39;mean_test_score&#39;]
mean_test_score = pd.DataFrame(scores, columns = [&#39;score&#39;])

res = pd.concat([hyper_parameters, mean_test_score], axis = 1)
print(res)


# ### 2. Random Search with Logistic Regression

# In[22]:


# 10. The logistic regression model
lr = LogisticRegression(penalty = &#39;elaticnet&#39;, solver = &#39;liblinear&#39;, max_iter = 1000)


# In[24]:


# 11. Define distributions to choose hyperparameters from
from scipy.stats import uniform
distributions = {&#39;penalty&#39;: [&#39;l1&#39;, &#39;l2&#39;], &#39;C&#39;: uniform(loc=0, scale = 100)}


# In[25]:


# 12. Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter = 8)

# Fit the random search model
clf.fit(X_train, y_train)


# In[26]:


# 13. Print best esimator and best score
best_score = clf.best_score_
test_score = clf.score(X_test, y_test)

# Print best score
print(best_score)

# Print the accuracy of the final model on the test data
print(test_score)


# In[27]:


#Print a table summarizing the results of RandomSearchCV
hyper_params = clf.cv_results_[&#39;params&#39;]
hyper_parameters = pd.DataFrame.from_dict(hyper_params)

scores = clf.cv_results_[&#39;mean_test_score&#39;]
mean_test_score = pd.DataFrame(scores, columns = [&#39;score&#39;])

res = pd.concat([hyper_parameters, mean_test_score], axis = 1)
print(res)


# In[ ]:




<script type="text/javascript" src="https://www.codecademy.com/assets/relay.js"></script></body></html>