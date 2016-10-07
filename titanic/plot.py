import pandas
import matplotlib.pyplot as plt
plt.style.use('ggplot')

ds = pandas.read_csv("data/train.csv")

# Info about columns and summary statistics
#ds.info()
#ds.describe()

ds_without_null_ages = ds[ds['Age'].notnull()]
ages_survived = ds_without_null_ages[ds_without_null_ages['Survived'] == 1]['Age']
ages_died = ds_without_null_ages[ds_without_null_ages['Survived'] == 0]['Age']
plt.hist(ages_survived.values, label="Survived", histtype='step', range=(0,100), bins=20)
plt.hist(ages_died.values, label="Died", histtype='step', range=(0,100), bins=20)
plt.xlabel("Age")
plt.ylabel("Entries")
plt.legend()
plt.show()

ds['Sex'] = ds['Sex'].map({'male': 0, 'female': 1})
plt.hist(ds[ds['Survived'] == 1]['Sex'].values, label="Survived", histtype='step', range=(0,2), bins=2)
plt.hist(ds[ds['Survived'] == 0]['Sex'].values, label="Died", histtype='step', range=(0,2), bins=2)
plt.xlabel("Gender")
plt.ylabel("Entries")
plt.legend()
plt.show()
