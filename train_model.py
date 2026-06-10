import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

DATA_PATH = "Salesdataset.csv"
MODEL_PATH = "best_voting_model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "model_columns.pkl"


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place Salesdataset.csv in the project root."
        )

    df = pd.read_csv(path)
    print(f"Loaded dataset with shape {df.shape}")
    return df


def preprocess_data(df):
    df = df.copy()

    # Convert date columns to datetime and extract features
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')

    df['signup_month'] = df['signup_date'].dt.month
    df['signup_day_of_week'] = df['signup_date'].dt.dayofweek
    df['last_purchase_month'] = df['last_purchase_date'].dt.month
    df['last_purchase_day_of_week'] = df['last_purchase_date'].dt.dayofweek
    df['customer_tenure_days'] = (df['last_purchase_date'] - df['signup_date']).dt.days

    df.drop(columns=['signup_date', 'last_purchase_date'], inplace=True, errors='ignore')

    # Convert churn target to numeric if necessary
    if df['churn'].dtype == 'object':
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(df['churn'])

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['customer_id', 'churn']]

    df_preprocessed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Data shape after one-hot encoding: {df_preprocessed.shape}")
    return df_preprocessed


def build_features(df_preprocessed):
    y = df_preprocessed['churn']
    X = df_preprocessed.drop(['churn', 'customer_id'], axis=1, errors='ignore')

    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()

    X = combined.drop(columns=['churn'])
    y = combined['churn']

    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    return X, y


def scale_features(X_train, X_test):
    original_numerical_features = [
        'age', 'total_visits', 'avg_session_time', 'pages_per_session',
        'email_open_rate', 'email_click_rate', 'total_spent', 'avg_order_value',
        'support_tickets', 'delivery_delay_days', 'satisfaction_score', 'nps_score',
        'marketing_spend_per_user', 'lifetime_value', 'last_3_month_purchase_freq'
    ]
    engineered_date_features = [
        'signup_month', 'signup_day_of_week', 'last_purchase_month',
        'last_purchase_day_of_week', 'customer_tenure_days'
    ]
    binary_features_not_to_scale = ['is_premium_user', 'discount_used', 'refund_requested']

    features_to_scale = [
        col for col in original_numerical_features + engineered_date_features
        if col in X_train.columns and col not in binary_features_not_to_scale
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    print(f"Scaled features: {features_to_scale}")
    return X_train_scaled, X_test_scaled, scaler


def train_best_model(X_train_scaled, y_train):
    estimators = [
        ('lr', LogisticRegression(random_state=42, solver='liblinear')),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True, random_state=42))
    ]

    voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

    param_grid = {
        'weights': [
            [1, 1, 1, 1],
            [2, 1, 1, 1],
            [1, 2, 1, 1],
            [1, 1, 2, 1],
            [1, 1, 1, 2],
            [2, 2, 1, 1],
            [1, 1, 2, 2],
            [2, 1, 1, 2]
        ]
    }

    grid_search = GridSearchCV(
        estimator=voting_clf,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train_scaled, y_train)

    print("Best voting classifier parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\nEvaluation results:")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")


def save_artifacts(model, scaler, model_columns):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model_columns, COLUMNS_PATH)
    print(f"Saved artifacts: {MODEL_PATH}, {SCALER_PATH}, {COLUMNS_PATH}")


def main():
    df = load_data(DATA_PATH)
    df_preprocessed = preprocess_data(df)
    X, y = build_features(df_preprocessed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    best_model = train_best_model(X_train_scaled, y_train)
    evaluate_model(best_model, X_test_scaled, y_test)

    model_columns = X.columns.tolist()
    save_artifacts(best_model, scaler, model_columns)


if __name__ == '__main__':
    main()
