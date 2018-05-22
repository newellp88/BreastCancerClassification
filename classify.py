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

data_file = 'cancer_data.csv'
graphs = 'graphs/'

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
"""
bars = sns.countplot(df['diagnosis'],label="Count")
bars = bars.get_figure()
bars.suptitle('Data visualization 1: Cancer, 0: Not cancer')
bars.savefig(graphs + 'data_viz.png')
bars.show()
"""
# feature selection, exploring correlation
corr = df[features_mean].corr()
"""
heat = sns.heatmap(corr, cbar=True, square=True, annot=True, linewidths=1.0,
				fmt='.2f', annot_kws={'size':15}, xticklabels=features_mean,
				yticklabels=features_mean, cmap='coolwarm')
heat = heat.get_figure()
heat.suptitle('Feature Heatmap')
heat.savefig(graphs + 'feature_heatmap.png')
heat.show()

cluster = sns.clustermap(corr, cmap='mako')
cluster.savefig(graphs + 'feature_clustermap.png')
plt.show()
"""
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
rf_acc = metrics.accuracy_score(prediction, test_y)
print("Random Forest accuracy: ", rf_acc)

# SVM model
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
svm_acc = metrics.accuracy_score(prediction, test_y)
print("SVM accuracy ", svm_acc)

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
rff_acc = metrics.accuracy_score(prediction, test_y)
print("All features used, Random Forest accuracy: ", rff_acc)

# make a barchart to compare the three classification models
# seaborn's barchart is a bit complicated, using matplotlib instead
acc = {'RandomForest':rf_acc, 'SVM':svm_acc, 'FullFeature':rff_acc}
title = 'Model comparision'
bar = plt.bar(range(len(acc)), list(acc.values()), align='center', color='g')
plt.xticks(range(len(acc)), list(acc.keys()))
plt.ylabel('Model accuracy')
plt.xlabel('Model name/type')
plt.title(title)
rects = bar.patches
for rec in rects:
	h = rec.get_height()
	label = ('%.2f' % (h * 100)) + '%'
	x = rec.get_x() + rec.get_width() / 2.0
	plt.annotate(label, (x,h), xytext=(0,10),
				textcoords='offset points',ha='center',va='top')
plt.savefig(graphs + '%s.png' % (title))
plt.show()
