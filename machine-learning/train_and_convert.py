import numpy as np
# Monkeypatch numpy to support old tensorflowjs imports on newer numpy
np.object = object
np.bool = bool

import pandas as pd
import json
import os
import tensorflow as tf

import sys
import types

# Mock tensorflow_hub entirely so tensorflowjs doesn't import the real one (which has compatibility issues)
hub_mock = types.ModuleType('tensorflow_hub')
sys.modules['tensorflow_hub'] = hub_mock
sys.modules['tensorflow_hub.estimator'] = types.ModuleType('tensorflow_hub.estimator')

import h5py
# Monkeypatch h5py attribute reader to return Keras version 2.15.0 to bypass tensorflowjs check
orig_get = h5py.AttributeManager.get
def patched_get(self, name, default=None):
    if name == 'keras_version':
        return '2.15.0'
    return orig_get(self, name, default)
h5py.AttributeManager.get = patched_get

orig_getitem = h5py.AttributeManager.__getitem__
def patched_getitem(self, name):
    if name == 'keras_version':
        return '2.15.0'
    return orig_getitem(self, name)
h5py.AttributeManager.__getitem__ = patched_getitem

import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_and_convert():
    # 1. Read data
    df = pd.read_csv('01ad36f0-a2e5-49c5-8665-524accb85f05-train.csv')
    
    # Drop EmployeeID if it exists
    if 'EmployeeID' in df.columns:
        df = df.drop(columns=['EmployeeID'])
        
    # 2. Preprocessing
    # Separate features and target
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    # Store mapping and scaling info
    preprocess_meta = {
        'features': list(X.columns),
        'categorical_mapping': {},
        'numerical_scaling': {}
    }
    
    # Process categorical and numerical with missing value handling
    for col in X.columns:
        if X[col].dtype == 'object':
            # Fill missing with mode
            X[col] = X[col].fillna(X[col].mode()[0])
            # It's categorical. Let's map unique values to integers
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            # Store mapping
            mapping = {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
            preprocess_meta['categorical_mapping'][col] = mapping
        else:
            # Fill missing with median
            X[col] = X[col].fillna(X[col].median())
            # It's numerical. Let's record mean and std for frontend scaling
            mean = float(X[col].mean())
            std = float(X[col].std())
            # Handle 0 std
            if std == 0:
                std = 1.0
            preprocess_meta['numerical_scaling'][col] = {'mean': mean, 'std': std}
            # Scale in python
            X[col] = (X[col] - mean) / std

    # Convert to numpy arrays
    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=np.float32)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
    
    # 3. Build Model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train Model
    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # 4. Save Model to web-app/public/model
    save_dir = os.path.join('.', 'web-app', 'public', 'model')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving tfjs model to {save_dir}")
    tfjs.converters.save_keras_model(model, save_dir)
    
    # 5. Save preprocess_meta.json
    meta_path = os.path.join(save_dir, 'preprocess_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(preprocess_meta, f, indent=4)
    print(f"Saved preprocess metadata to {meta_path}")

    # 6. Patch model.json to fix Keras 3 compatibility with TFJS
    model_json_path = os.path.join(save_dir, 'model.json')
    if os.path.exists(model_json_path):
        with open(model_json_path, 'r') as f:
            model_data = f.read()
        model_data = model_data.replace('"batch_shape":', '"batchInputShape":')
        model_data = model_data.replace('"sequential/', '"')
        with open(model_json_path, 'w') as f:
            f.write(model_data)
        print("Patched model.json for TFJS compatibility.")

if __name__ == '__main__':
    train_and_convert()
