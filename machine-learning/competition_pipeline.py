import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import os

def run_competition_pipeline():
    print("Memuat dataset kompetisi...")
    # Load actual data
    train = pd.read_csv('01ad36f0-a2e5-49c5-8665-524accb85f05-train.csv')
    test = pd.read_csv('23c4d140-9de4-46d2-9d6c-fc1c75df5d9c-test.csv')
    
    # Simpan EmployeeID untuk file submission
    test_ids = test['EmployeeID']
    
    # Preprocessing
    # Gabungkan data untuk encoding yang konsisten
    train['is_train'] = 1
    test['is_train'] = 0
    test['Attrition'] = -1
    
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop(columns=['EmployeeID'])
    
    # Menangani Nilai Hilang (Missing Values)
    for col in combined.columns:
        if combined[col].dtype == 'object':
            combined[col] = combined[col].fillna(combined[col].mode()[0])
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
        else:
            combined[col] = combined[col].fillna(combined[col].median())
            
    # Pisahkan kembali
    df_train = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    df_test = combined[combined['is_train'] == 0].drop(columns=['is_train', 'Attrition'])
    
    X = df_train.drop(columns=['Attrition'])
    y = df_train['Attrition'].astype(int)
    
    # Hitung rasio untuk imbalanced class (scale_pos_weight)
    stay_count = (y == 0).sum()
    leave_count = (y == 1).sum()
    spw = stay_count / leave_count
    print(f"Distribusi Kelas - Stay: {stay_count}, Leave: {leave_count}. Scale_pos_weight: {spw:.2f}")
    
    # Model Setup (XGBoost) khusus dirancang memaksimalkan F1
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw, # Sangat krusial untuk imbalanced dataset!
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Evaluasi Lokal menggunakan Cross Validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"Estimasi F1 Score (Cross-Validation): {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Latih model final di seluruh data training
    print("Melatih model final...")
    model.fit(X, y)
    
    # Prediksi data uji
    print("Melakukan prediksi pada test.csv...")
    preds = model.predict(df_test)
    
    # Buat file submission
    submission = pd.DataFrame({
        'employee_id': test_ids,
        'attrition': preds
    })
    
    sub_path = 'HR_sample_submission.csv'
    submission.to_csv(sub_path, index=False)
    print(f"Sukses! File {sub_path} telah dibuat dan siap untuk disubmit ke kompetisi.")

if __name__ == "__main__":
    run_competition_pipeline()
