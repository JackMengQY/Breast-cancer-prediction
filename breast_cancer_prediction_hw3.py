from __future__ import division, print_function, unicode_literals
import pandas as pd
import numpy as np
from nltk.classify import decisiontree
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
data = pd.read_csv(url, header=None)
print(data.columns)

np.random.seed(42)

# explore the data ()
print("Data statistics:")
print(data.describe())

data_correlation = data.select_dtypes(include=[np.number])
print("Correlation matrix:")
print(data_correlation.corr())

# get target and training attribute
X = data.iloc[:, 2:32].values
y = data.iloc[:, 1].values

# partition into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# find the optimal parameters using grid search
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
# GridSearchCV performs an exhaustive search over specified parameter values for an estimator
# The parameters of the estimator used to apply these methods are optimized by cross-validated
# grid-search over a parameter grid.
# Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, recall_score
# Pipeline of transforms with a final estimator
from sklearn.pipeline import Pipeline

inner_cv = KFold(n_splits=5, shuffle=True) # inner cross-validation folds preferences
outer_cv = KFold(n_splits=5, shuffle=True) # outer cross-validation folds preferences
f1_scorer = make_scorer(f1_score, pos_label='M')
# Choosing depth of the tree AND splitting criterion AND min_samples_leaf AND min_samples_split
gs_dt2 = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7,8,9, None],
                               'criterion':['gini','entropy'],
                               'min_samples_leaf':[1,2,3,4,5,6,7],
                               'min_samples_split':[2,3,4,5,6,7]}],
                  scoring=f1_scorer,
                  cv=inner_cv,
                  n_jobs=4)

gs_dt2 = gs_dt2.fit(X,y)
print("\n Parameter Tuning - decision tree")
print("Non-nested CV f1: ", gs_dt2.best_score_)
print("Optimal Parameter: ", gs_dt2.best_params_)
print("Optimal Estimator: ", gs_dt2.best_estimator_)
nested_score_gs_dt2 = cross_val_score(gs_dt2, X=X, y=y, cv=outer_cv,scoring=f1_scorer)
print("Nested CV f1: ",nested_score_gs_dt2.mean(), " +/- ", nested_score_gs_dt2.std())





# build decision tree based on optimal parameters
clf3 = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=6, min_samples_split=2)
y_pred = clf3.fit(X_train, y_train).predict(X_test)

# Accuracy
print('Decision tree accuracy (out-of-sample): %.2f' % accuracy_score(y_test,
                                                        y_pred))  # Accuracy is calculated and printed for both the test (out-of-sample) dataset.
#print('Accuracy (in-sample): %.2f' % accuracy_score(y_train,
#                                                   y_pred_insample))  # Accuracy is calculated and printed for both the training (in-sample) dataset.

# confustion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # devide absolute number of observations with sum across columns to get the relative percentage of observations
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)         # shows the confusion matrix in the console
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))                       # add tick marks to the confusion matrix
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'                          # choose format depending on whether the confusion matrix is normalized or not
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # loop that adds the value to each cell of the confusion matrix
        plt.text(j, i, format(cm[i, j], fmt),                              # we reformat how the cell values are displayed accroding to the variable fmt we defined before
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix to evaluate the accuracy of a classification
cnf_matrix = confusion_matrix(y_test, y_pred)
#Determine the way floating point numbers are displayed
np.set_printoptions(precision=2)                              # number of digits of precision for floating point output

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      title='Decision tree confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      normalize=True,
                      title='Decision tree normalized confusion matrix')

plt.show()
# precision
print('Decision tree-Precision (out-of-sample): %.2f' % precision_score(y_test, y_pred, pos_label='M'))
# recall
print('Decision tree-Recall (out-of-sample): %.2f' % recall_score(y_test,y_pred, pos_label='M'))

# F1 score
# The F1 score is a harmonic mean of precision and recall, providing a balance between the two metrics
print('F1 score decision tree (out-of-sample): ', f1_score(y_test, y_pred,
                                             average='macro'))  # average='macro' calculate metrics for each label, and find their unweighted mean
#print('F1 score (in-sample)    : ', f1_score(y_train, y_pred_insample,
#                                            average='macro'))  # The average='macro' argument calculates the metric independently for each class and then takes the average, not considering label imbalance.

# Kappa score
# Cohen's Kappa score measures the agreement between the predictions and the actual values, accounting for the possibility of agreement occurring by chance
# It's especially useful when the classes are imbalanced
print('Kappa score decision tree (out-of-sample): ',
      cohen_kappa_score(y_test, y_pred))  # computes Cohen’s kappa: a statistic that measures inter-annotator agreement
#print('Kappa score (in-sample)    : ', cohen_kappa_score(y_train,
#                                                        y_pred_insample))  # (i.e., agreement between predictions and actual values of target variables)

# Build a text report showing the main classification metrics (out-of-sample performance)
# classification_report function provides a comprehensive report displaying key metrics
print(classification_report(y_test, y_pred,
                            target_names= ['M', 'B']))  # builds a text report showing the main classification metrics (precision, recall, f1-score)




# kNN - find optimal parameters
# Normalization for Knn
from sklearn.pipeline import Pipeline
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

pipe = Pipeline([
        ('sc', StandardScaler()),
        ('knn', KNeighborsClassifier(p=2,
                                     metric='minkowski'))
      ])

f1_scorer = make_scorer(f1_score, pos_label='M')
params = {
        'knn__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
        'knn__weights': ['uniform', 'distance']
    }

gs_knn2 = GridSearchCV(estimator=pipe,
                  param_grid=params,
                  scoring=f1_scorer,
                  cv=inner_cv,
                  n_jobs=4)

gs_knn2 = gs_knn2.fit(X,y)
print("\n Parameter Tuning kNN")
print("Non-nested CV f1: ", gs_knn2.best_score_)
print("Optimal Parameter: ", gs_knn2.best_params_)
print("Optimal Estimator: ", gs_knn2.best_estimator_) # Estimator that was chosen by the search, i.e. estimator which gave highest score
nested_score_gs_knn2 = cross_val_score(gs_knn2, X=X, y=y, cv=outer_cv, scoring=f1_scorer)
print("Nested CV f1: ",nested_score_gs_knn2.mean(), " +/- ", nested_score_gs_knn2.std())




# train the knn based on optimal parameters
knn = neighbors.KNeighborsClassifier(n_neighbors=9,
                                     p=2, metric='euclidean',
                                     n_jobs=4,
                                     weights='uniform')
knn = knn.fit(X_train_std, y_train)

# model evaluation
y_pred = knn.predict(X_test_std)


# Accuracy
print('kNN accuracy (out-of-sample): %.2f' % accuracy_score(y_test,
                                                        y_pred))  # Accuracy is calculated and printed for both the test (out-of-sample) dataset.
#print('Accuracy (in-sample): %.2f' % accuracy_score(y_train,
#                                                   y_pred_insample))  # Accuracy is calculated and printed for both the training (in-sample) dataset.

# confustion matrix
import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # devide absolute number of observations with sum across columns to get the relative percentage of observations
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)         # shows the confusion matrix in the console
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))                       # add tick marks to the confusion matrix
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'                          # choose format depending on whether the confusion matrix is normalized or not
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # loop that adds the value to each cell of the confusion matrix
        plt.text(j, i, format(cm[i, j], fmt),                              # we reformat how the cell values are displayed accroding to the variable fmt we defined before
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix to evaluate the accuracy of a classification
cnf_matrix = confusion_matrix(y_test, y_pred)
#Determine the way floating point numbers are displayed
np.set_printoptions(precision=2)                              # number of digits of precision for floating point output

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# precision
print('Precision (out-of-sample): %.2f' % precision_score(y_test, y_pred, pos_label='M'))
# recall
print('Recall (out-of-sample): %.2f' % recall_score(y_test,y_pred, pos_label='M'))

# F1 score
# The F1 score is a harmonic mean of precision and recall, providing a balance between the two metrics
print('F1 score (out-of-sample): ', f1_score(y_test, y_pred,
                                             average='macro'))  # average='macro' calculate metrics for each label, and find their unweighted mean
#print('F1 score (in-sample)    : ', f1_score(y_train, y_pred_insample,
#                                            average='macro'))  # The average='macro' argument calculates the metric independently for each class and then takes the average, not considering label imbalance.

# Kappa score
# Cohen's Kappa score measures the agreement between the predictions and the actual values, accounting for the possibility of agreement occurring by chance
# It's especially useful when the classes are imbalanced
print('Kappa score (out-of-sample): ',
      cohen_kappa_score(y_test, y_pred))  # computes Cohen’s kappa: a statistic that measures inter-annotator agreement
#print('Kappa score (in-sample)    : ', cohen_kappa_score(y_train,
#                                                        y_pred_insample))  # (i.e., agreement between predictions and actual values of target variables)

# Build a text report showing the main classification metrics (out-of-sample performance)
# classification_report function provides a comprehensive report displaying key metrics
print(classification_report(y_test, y_pred,
                            target_names= ['M', 'B']))  # builds a text report showing the main classification metrics (precision, recall, f1-score)



# for logistic regression


# Import necessary libraries and modules
from sklearn import linear_model
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# Split the data into training and testing
X_train_lg, X_test_lg, y_train_lg, y_test_lg = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# find optimal parameter
f1_scorer = make_scorer(f1_score, pos_label='M')
gs_lr2 = GridSearchCV(estimator=LogisticRegression(random_state=42, solver='liblinear'),
                  param_grid=[{'C': [ 0.00001, 0.0001, 0.001, 0.01, 0.1 ,1 ,10 ,100, 1000, 10000, 100000, 1000000, 10000000],
                              'penalty':['l1','l2']}],
                  scoring=f1_scorer,
                  cv=inner_cv)

gs_lr2 = gs_lr2.fit(X,y)
print("\n Parameter Tuning - Logistic Regression")
print("Non-nested CV f1: ", gs_lr2.best_score_)
print("Optimal Parameter: ", gs_lr2.best_params_)
print("Optimal Estimator: ", gs_lr2.best_estimator_)
nested_score_gs_lr2 = cross_val_score(gs_lr2, X=X, y=y, cv=outer_cv, scoring=f1_scorer)
print("Nested CV f1:",nested_score_gs_lr2.mean(), " +/- ", nested_score_gs_lr2.std())


# train the logistic model with optimal parameters
clf = LogisticRegression(C=1000, penalty='l1', solver='liblinear', random_state=42, max_iter=60)
clf.fit(X_train_lg, y_train_lg)

print('The weights of the attributes are:', clf.coef_)
print('The weights of the intercepts are:', clf.intercept_)

# apply the trained model to test set
y_pred = clf.predict(X_test_lg)             # generate classification prediction and store them in y_pred
                                         # in scikit-learn's LogisticRegression the default threshold for the .predict() method is 0.5
y_pred_prob = clf.predict_proba(X_test_lg)  # estimate class probabilities

# Print the first elements of the arrays containing predictions, predicted class probabilities,
# and the sum of predicted probabilities for the first test sample
print('The predictions are:', y_pred[0], y_pred_prob[0], np.sum(y_pred_prob[0])) # prints first elements of arrays

# evaluation
# confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # devide absolute number of observations with sum across columns to get the relative percentage of observations
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)         # shows the confusion matrix in the console
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))                       # add tick marks to the confusion matrix
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'                          # choose format depending on whether the confusion matrix is normalized or not
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): # loop that adds the value to each cell of the confusion matrix
        plt.text(j, i, format(cm[i, j], fmt),                              # we reformat how the cell values are displayed accroding to the variable fmt we defined before
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix to evaluate the accuracy of a classification
cnf_matrix = confusion_matrix(y_test_lg, y_pred)
#Determine the way floating point numbers are displayed
np.set_printoptions(precision=2)                              # number of digits of precision for floating point output

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=['M', 'B'],
                      normalize=True,
                      title='Normalized confusion matrix')

plt.show()
# precision
print('Accuracy (out-of-sample): %.2f' % accuracy_score(y_test_lg, y_pred))
print('Precision (out-of-sample): %.2f' % precision_score(y_test_lg, y_pred, pos_label='M'))
# recall
print('Recall (out-of-sample): %.2f' % recall_score(y_test_lg,y_pred, pos_label='M'))
print('f1-score (out-of-sample): %.2f' % f1_score(y_test_lg,y_pred, pos_label='M'))
print(classification_report(y_test_lg, y_pred, target_names=['M','B']))










