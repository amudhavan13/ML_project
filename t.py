from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load the Iris dataset (example)
iris = pd.read_csv('heart.csv')
print(iris)
x=iris.loc[:, 'age':'thal']
y=iris['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(criterion="entropy")

dt.fit(X_train, y_train)

joblib.dump(dt, 'model1.joblib')