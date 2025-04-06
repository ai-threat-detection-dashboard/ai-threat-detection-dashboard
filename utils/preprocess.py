import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_logs(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['login_fail'] = df['status'].apply(lambda x: 1 if x == 'FAIL' else 0)

    features = df[['hour', 'login_fail']].groupby('hour').sum().reset_index()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features[['hour', 'login_fail']])

    return features, scaled_features
  
