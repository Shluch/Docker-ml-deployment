import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'model.joblib')
