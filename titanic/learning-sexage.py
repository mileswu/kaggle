# Options
#features = ['Sex', 'Age', 'Pclass', 'Fare']
features = ['Sex', 'Age', 'Pclass', 'Fare']
method = "nn-mlp"

# Get data
import pandas
ds = pandas.read_csv("data/train.csv")
ds_test = pandas.read_csv("data/test.csv")

# Clean age and convert sex
ds.loc[ds['Age'].isnull(), 'Age'] = ds['Age'].median()
ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1})
ds_test.loc[ds_test['Age'].isnull(), 'Age'] = ds['Age'].median()
ds_test['Sex'] = ds_test['Sex'].map({'male': 0, 'female': 1})
ds.loc[ds['Fare'].isnull(), 'Fare'] = ds['Fare'].median()
ds_test.loc[ds_test['Fare'].isnull(), 'Fare'] = ds['Fare'].median()

ds['Married'] = 0
ds.loc[(ds['Sex'] == 1) & (ds['Name'].str.contains("Mrs")), 'Married'] = 1
ds.loc[ds['Sex'] == 0, 'Married'] = 2
ds_test['Married'] = 0
ds_test.loc[(ds_test['Sex'] == 1) & (ds_test['Name'].str.contains("Mrs")), 'Married'] = 1
ds_test.loc[ds_test['Sex'] == 0, 'Married'] = 2

ds['NotMr'] = 1
ds.loc[(ds['Sex'] == 0) & (ds['Name'].str.contains("Mr")), 'NotMr'] = 0
ds.loc[ds['Sex'] == 1, 'NotMr'] = 3
ds_test['NotMr'] = 1
ds_test.loc[(ds['Sex'] == 0) & (ds_test['Name'].str.contains("Mr")), 'NotMr'] = 0
ds_test.loc[ds_test['Sex'] == 1, 'NotMr'] = 3

ds['FamilySize'] = ds['Parch'] + ds['SibSp']
ds_test['FamilySize'] = ds_test['Parch'] + ds_test['SibSp']

print("Survival Rate for male v female: %f v %f" % (ds[ds['Sex'] == 0]['Survived'].mean(), ds[ds['Sex'] == 1]['Survived'].mean()))
print("Survival Rate for 1st/2nd/3rd class: %f v %f v %f" % (ds[ds['Pclass'] == 1]['Survived'].mean(), ds[ds['Pclass'] == 2]['Survived'].mean(), ds[ds['Pclass'] == 3]['Survived'].mean()))
print("Survival Rate for females with Mrs v. NotMrs: %f v %f" % (ds[(ds['Sex'] == 1) & (ds['Married'] == 1)]['Survived'].mean(), ds[(ds['Sex'] == 1) & (ds['Married'] == 0)]['Survived'].mean()))
print("Survival Rate for males with Mr v. NotMr: %f v %f" % (ds[(ds['Sex'] == 0) & (ds['NotMr'] == 0)]['Survived'].mean(), ds[(ds['Sex'] == 0) & (ds['NotMr'] == 1)]['Survived'].mean()))

# Check that there is no missing data
if ds[features].isnull().sum().sum() !=0 or ds_test[features].isnull().sum().sum() !=0:
    print("WARNING: there is null data")
    print(ds[features].isnull().sum())
    print(ds_test[features].isnull().sum())

# Get raw numpy arrays for learning
raw_training = ds[['Survived'] + features].values
raw_testing = ds_test[features].values

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
elif method == "svm-svc":
    from sklearn.svm import SVC
    classifier = SVC(random_state=0)
elif method == "logistic":
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
elif method == "logistic-cv":
    from sklearn.linear_model import LogisticRegressionCV
    classifier = LogisticRegressionCV(Cs=100, random_state=0)
elif method == "nn-mlp":
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), random_state=0)

classifier.fit(raw_training[0::, 1::], raw_training[0::, 0])

print("Predicted correctly on training data: %f" % classifier.score(raw_training[0::, 1::], raw_training[0::, 0]))
if method == "tree" or method == "randomforest" or method == "gradientboostedtree":
    print("Feature importances:")
    for (i, j) in zip(features, classifier.feature_importances_):
        print("  %s - %f" % (i, j))
elif method == "logistic" or method == "logistic-cv":
    print("Feature coefficients:")
    for (i, j) in zip(features, classifier.coef_[0]):
        print("  %s - %f" % (i, j))

# Do predicting
raw_predictions = classifier.predict(raw_testing).astype(int)
ds_test['Survived'] = raw_predictions
output_filename = "learning-sexage-" + method + ".csv"
ds_test[['PassengerId', 'Survived']].to_csv(output_filename, index=False)
