import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def train_and_save():
    # ── Load Data ──────────────────────────
    df = pd.read_csv('heart.csv')
    
    # ── Preprocessing ──────────────────────
    le = LabelEncoder()
    categorical_cols = ['Sex', 'ChestPainType', 
                       'RestingECG', 'ExerciseAngina', 'ST_Slope']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # ── Train/Test Split ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ── Scaling ────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Model Training ─────────────────────
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # ── Evaluation ─────────────────────────
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, 
                        model.predict_proba(X_test_scaled)[:,1])
    
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ AUC Score: {auc:.3f}")
    print(classification_report(y_test, y_pred))
    
    # ── Save Model ─────────────────────────
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns.tolist(), 'features.pkl')
    print("✅ Model saved!")

if __name__ == '__main__':
    train_and_save()
    