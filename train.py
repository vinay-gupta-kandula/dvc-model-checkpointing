import json
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # 2. Define and train model (v1.0 baseline: n_estimators=10)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # 4. Save artifacts
    joblib.dump(model, 'model.joblib')
    
    with open('metrics.json', 'w') as f:
        json.dump({'accuracy': accuracy}, f)
    
    print(f"Training complete! Model accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
