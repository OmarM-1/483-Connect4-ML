from sklearn.ensemble import RandomForestClassifier
from dataset import load_train_val_with_mask
from evaluate import evaluate_predictions, print_metrics

def main():
    X_train, y_train, train_masks, X_val, y_val, val_masks = load_train_val_with_mask()
    model = RandomForestClassifier(criterion = 'entropy', max_depth=16, random_state = 1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)

    metrics = evaluate_predictions(
        y_true = y_val,
        y_pred = y_pred,
        y_score = y_score,
        legal_masks = val_masks
    )

    print_metrics(metrics)

if __name__ == "__main__":
    main()