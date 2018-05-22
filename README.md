# Introduction

Data from an old ["Kaggle.com"](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) competition.

Exploratory machine learning/classification set of breast cancer data. The data itself is pretty straightforward, roughly 2/3 of the patients don't have cancer and about 1/3 do.

![data vis](/graphs/data_viz.png)

When we look at the features, we can see that there isn't a lot of correlation between the inputs of this dataset which explains why machine learning might be useful in the medical industry.

![clustermap](/graphs/feature_clustermap.png)

The clustermap highlights just a few 'hot' areas of high positive correlation and three 'cool' areas of strong negative correlation. Otherwise, we've got swimming temperatures.

## Setting up the models

scikit learn models tend to be straightforward, the important part is to make sure the data is prepared properly. In this example, we do that by eliminating columns that don't contribute to our solution:

```
# clean up the dataframe
df = pd.read_csv(data_file)
df.drop("Unnamed: 32", axis = 1, inplace = True)
df.drop('id', axis = 1, inplace = True) #axis=1 means the whole column
```
This leaves us with only the diagnosis (column 0) and the input data. Next, we'll set our input variables and split our train/test data using a semi-random selection of features from the 'mean' category:

```
# set and split prediction variables
prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean']
train, test = train_test_split(df, test_size=0.3)

train_X = train[prediction_var]
train_y = train.diagnosis

test_X = test[prediction_var]
test_y = test.diagnosis
```

Then we define, fit, predict and score our sklearn models which tend to follow the same formula:

```
# build some models using just the 5 features selected before
model = RandomForestClassifier(n_estimators = 100)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
rf_acc = metrics.accuracy_score(prediction, test_y)
```

Finally, we will want to know how our models fared in general and against each other:


![model comparison](/graphs/Model_comparision.png)


Not too bad. Although, we should probably test more input data and features before we get too excited.


