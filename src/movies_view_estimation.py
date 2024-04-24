import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data."""
    df['Year'] = df['Year'].astype(float)
    df['Rating'] = df['Rating'].astype(float)
    df['Rating Count'] = df['Rating Count'].str.replace(",", "").astype(float)
    return df

def split_data(df):
    """Split data into train and test sets."""
    train_data = df[df['Label'].notnull()]
    test_data = df[df['Label'].isnull()]
    return train_data, test_data

def vectorize_text(train_texts, test_texts):
    """Vectorize text data."""
    cv = CountVectorizer()
    train_texts = cv.fit_transform(train_texts)
    test_texts = cv.transform(test_texts)
    return train_texts, test_texts

def scale_features(X_train, X_test):
    """Scale numerical features."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler

def scale_labels(y_train):
    """Scale labels."""
    label_scaler = StandardScaler()
    y_train = label_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    return y_train, label_scaler

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    classifier = XGBRegressor()
    classifier.fit(X_train, y_train)
    return classifier

def predict(model, X_test, label_scaler):
    """Predict movie views using the trained model."""
    y_pred = model.predict(X_test)
    y_test = label_scaler.inverse_transform(y_pred)
    return y_test

def main():
    # Load data
    df = load_data("data/imdb-tv-ratings/top-250-movie-ratings-label.csv")
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data
    train_data, test_data = split_data(df)
    
    # Extract features and labels
    train_texts = train_data['Title'].str.lower()
    test_texts = test_data['Title'].str.lower()
    X_train = train_data[['Year', 'Rating', 'Rating Count']]
    y_train = train_data['Label'].values
    
    # Vectorize text data
    X_train_text, X_test_text = vectorize_text(train_texts, test_texts)
    
    # Scale numerical features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, test_data[['Year', 'Rating', 'Rating Count']])
    
    # Scale labels
    y_train_scaled, label_scaler = scale_labels(y_train)
    
    # Train the model
    model = train_model(X_train_scaled, y_train_scaled)
    
    # Make predictions
    predictions = predict(model, X_test_scaled, label_scaler)
    
    # Output predictions
    print(predictions)

if __name__ == "__main__":
    main()
