from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

def mlp_train_and_predict(X_train, X_test, y_train):
    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # MLP Regressor setup
    mlp = MLPRegressor(random_state=12345, early_stopping=True, n_iter_no_change=50)

    # Grid search for hyperparameters
    param_grid = {
        'hidden_layer_sizes': [(4,), (64,), (128,), (256,), (512,)],
        'max_iter': [100, 1000, 10000]
    }
    
    grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    # Best model
    best_mlp = grid.best_estimator_

    # Train final model
    best_mlp.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = best_mlp.predict(X_test_scaled)
    return y_pred, grid.best_params_