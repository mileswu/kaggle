import pandas

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

# Choose learning method
method = "tree"

# Do learning
if method == "tree":
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state=0)
elif method == "randomforest":
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, random_state=0)
elif method == "gradientboostedtree":
    from sklearn.ensemble import GradientBoostingClassifier
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
classifier.fit(raw_training[0::, 1::], raw_training[0::, 0])
print("Predicted correctly on training data: %f" % classifier.score(raw_training[0::, 1::], raw_training[0::, 0]))
raw_predictions = classifier.predict(raw_testing).astype(int)

# Prepare output
ds_test['Survived'] = raw_predictions
output_filename = "learning-sexage-" + method + ".csv"
ds_test[['PassengerId', 'Survived']].to_csv(output_filename, index=False)
