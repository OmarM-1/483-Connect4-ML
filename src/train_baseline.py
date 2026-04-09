
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataset import load_train_val

def main():
    X_train, y_train, X_val, y_val = load_train_val()
    model = LogisticRegression(
        max_iter=200,
        verbose=1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)

    print("\nValidation Accuracy:", acc)


if __name__ == "__main__":
    main()
