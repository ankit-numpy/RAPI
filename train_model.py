from sklearn. datasets import load_iris
data = load_iris()
X, y = data.data, data.target # pyright: ignore[reportAttributeAccessIssue]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
import joblib
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved as 'iris_model.pkl'")