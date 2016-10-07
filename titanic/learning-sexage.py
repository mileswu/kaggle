import pandas
from sklearn.tree import DecisionTreeClassifier

ds = pandas.read_csv("data/train.csv")
ds_test = pandas.read_csv("data/test.csv")

# Clean age and convert sex
ds.loc[ds['Age'].isnull(), 'Age'] = -1
ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1})
ds_test.loc[ds_test['Age'].isnull(), 'Age'] = -1
ds_test['Sex'] = ds_test['Sex'].map({'male': 0, 'female': 1})

# Get raw numpy arrays for learning
raw_training = ds[['Survived', 'Sex', 'Age']].values
raw_testing = ds_test[['Sex', 'Age']].values

# Do learning
tree_classifier = DecisionTreeClassifier(random_state=0)
tree_classifier = DecisionTreeClassifier(random_state=0)

tree_classifier.fit(raw_training[0::, 1::], raw_training[0::, 0])
print("Predicted correctly on training data: %f" % tree_classifier.score(raw_training[0::, 1::], raw_training[0::, 0]))
raw_predictions = tree_classifier.predict(raw_testing).astype(int)

# Prepare output
ds_test['Survived'] = raw_predictions
ds_test[['PassengerId', 'Survived']].to_csv("tree.csv", index=False)
