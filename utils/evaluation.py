def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance."""
    print("\n--- Model Evaluation ---")

    c_index_train = model.score(X_train, y_train)
    c_index_test = model.score(X_test, y_test)

    print(f"Concordance Index (Training): {c_index_train:.4f}")
    print(f"Concordance Index (Testing):  {c_index_test:.4f}")
    print("--- End of Evaluation ---")

    return {
        'train_c_index': c_index_train,
        'test_c_index': c_index_test
    }