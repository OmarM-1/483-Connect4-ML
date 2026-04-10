from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import load_train_val

def main():
    X_train, y_train, X_val, y_val = load_train_val()
    model = RandomForestClassifier(criterion = 'entropy', max_depth=6)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    accuracy = accuracy_score(y_val, preds)

    print("\nValidation Accuracy:", accuracy)

if __name__ == "__main__":
    main()