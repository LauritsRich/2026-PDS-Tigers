from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import joblib




def svm_model_training(dev_x, dev_y):

    ### SVM Model Training
    # Pipeline (scaling + SVM)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])

    # Parameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto', 0.01, 0.001]
    }

    # GridSearchCV---> does the cross-validation, with the train/validation split
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    # Train
    grid_search.fit(dev_x, dev_y)

    # --- RESULTS ---
    print("Best Params:", grid_search.best_params_)
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Test evaluation
    SVM_model = grid_search.best_estimator_


    return SVM_model




def knn_model_training(dev_x, dev_y):

    ### KNN Model Training
    # Pipeline (scaling + KNN)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Parameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 11, 13, 15, 17, 19],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    }

    # GridSearchCV---> does the cross-validation, with the train/validation split
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    # Train
    grid_search.fit(dev_x, dev_y)

    # --- RESULTS ---
    print("Best Params:", grid_search.best_params_)
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # Test evaluation
    KNN_model = grid_search.best_estimator_


    return KNN_model



def main(features_path, prediction_results_path, base_model_path, load_model, model_type, extended_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param base_model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    :param model_type: String specifying the type of the model used for predictions. (KNN or SVM)
    :param extended_model: Boolean to use the extended features dataset if True, otherwise use the baseline features
    """

    baseline_features = ['asymmetry','compactness','convexity','r_var','g_var','b_var', 'h_var', 's_var', 'v_var', 'cancerous']
    extended_features = ['asymmetry','compactness','convexity', 'as_value', 'as_var', 
                         'mean_angle_h', 's_value', 'v_value', 'r_value', 'bs_var', 'h_var', 
                         's_var', 'circular_max_min_h', 'diameter', 'lacunarity', 'contrast', 'cancerous']
    # Select correct model path
    model_path = f"{base_model_path}_{model_type}_{'extended' if extended_model else 'baseline'}.pkl"
    prediction_path = f"{prediction_results_path}_{model_type}_{'extended' if extended_model else 'baseline'}.csv"
    # load dataset CSV file
    features_df = pd.read_csv(features_path)
    selected_features = features_df[extended_features if extended_model else baseline_features]
    selected_features = selected_features.dropna()

    x = selected_features.copy().drop('cancerous', axis = 1)
    y = selected_features['cancerous'].copy()

    # split the dataset into training and testing sets.
    dev_x, test_x, dev_y, test_y = train_test_split(
    x, y, stratify=y, random_state=0, test_size=0.2)

    # Load or train model
    if load_model:
        model = joblib.load(model_path)
    else:
        if model_type == 'SVM':
            model = svm_model_training(dev_x, dev_y)
        elif model_type == 'KNN':
            model = knn_model_training(dev_x, dev_y)
        else:
            print('model should be either knn or svm')

        joblib.dump(model, model_path)

    # test the classifier.
    y_probs = model.predict_proba(test_x)[:, 1]

    test_auc = roc_auc_score(test_y, y_probs)
    print(f"{model_type}TEST AUC:", test_auc)
    # write test results to CSV.
    results_df = pd.DataFrame({'probability': y_probs}, index=test_x.index)
    results_df.to_csv(prediction_path)



if __name__ == "__main__":
    features_path = "2026-PDS-Tigers/data/features.csv"
    prediction_results_path = "2026-PDS-Tigers/results/predictions/predictions"
    base_model_path = "2026-PDS-Tigers/results/models/model"
    load_model = False
    model_type = 'SVM'
    extended_model = False

    main(features_path, prediction_results_path, base_model_path, load_model, model_type, extended_model)