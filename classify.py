#! Classifying tumors as malignant or benign from Univ Wisc data

# import standard data science and graphing modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import some sklearn libraries for prediction and classification testing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import svm

data_file = 'breast_data.csv'

# clean up the dataframe
df = pd.read_csv(data_file)
df.drop("Unnamed: 32", axis = 1, inplace = True)
df.drop('id', axis = 1, inplace = True) #axis=1 means the whole column

# divide the data into it's three general categories
features_mean = list(df.columns[1:11])
features_se = list(df.columns[12:20])
features_worst = list(df.columns[21:31])

# chart diagnosis; M: malignant, B: benign
df["diagnosis"] = df["diagnosis"].map({'M':1,'B':0})

# explore the data now
bars = sns.countplot(df['diagnosis'],label="Count")
bars = bars.get_figure()
bars.suptitle('Is it breast cancer?')
bars.savefig('breat_cancer_classification.png')
bars.show()

# feature selection, exploring correlation
corr = df[features_mean].corr()
heat = sns.heatmap(corr, cbar=True, square=True, annot=True,
				fmt='.2f', annot_kws={'size':15}, xticklabels=features_mean,
				yticklabels=features_mean, cmap='coolwarm')
heat = heat.get_figure()
heat.suptitle('Feature Heatmap')
heat.savefig('feature_heatmap.png')
heat.show()

"""
Observations:
	1. the radius, perimeter, and area are highly correlated as expected fromtheir relation. From the we can use any one of them as a feature
	2. compactness_mean, concavity_mean, and concavepoint_mean are also highly correlated, we'll use compactness_mean
"""
# set and split prediction variables
prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
train, test = train_test_split(df, test_size=0.3)

train_X = train[prediction_var]
train_y = train.diagnosis

test_X = test[prediction_var]
test_y = test.diagnosis

# build some models using just the 5 features selected before
model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("Random Forest accuracy: ", metrics.accuracy_score(prediction, test_y))

# SVM model
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("SVM accuracy ", metrics.accuracy_score(prediction, test_y))

# build models using all the features
prediction_var = features_mean
train_X = train[prediction_var]
train_y = train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

# Random forest model
model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print("All features used, Random Forest accuracy: ", metrics.accuracy_score(prediction, test_y))
