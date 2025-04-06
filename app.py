from flask import Flask, render_template, request
import joblib
from utils.preprocess import preprocess_logs

app = Flask(__name__)
model = joblib.load("model/anomaly_detector.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['logfile']
    path = "logs/uploaded.csv"
    file.save(path)
    features, X = preprocess_logs(path)
    preds = model.predict(X)
    features['anomaly'] = preds
    return render_template('dashboard.html', data=features.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
