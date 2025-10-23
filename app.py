from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


app = Flask(__name__)

model = load_model("stock_cnn.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def transform_data(look_back, stock_data):
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.droplevel(1)
        stock_data.columns.name = None

    prices = stock_data.copy()
    prices['Date'] = prices.index
    prices = prices.dropna()

    prices[['Open','High','Low','Close']] = prices[['Open','High','Low','Close']].astype(float)
    prices['Volume'] = prices['Volume'].astype(int)

    prices[['Open_pc','High_pc','Low_pc','Close_pc']] = prices[['Open','High','Low','Close']].pct_change()
    features = ['Open_pc','High_pc','Low_pc','Close_pc']

    for num in range(look_back):
        for price_type in features:
            col_name = f"{price_type}_{num}"
            prices[col_name] = prices[price_type].shift(num + 1)

    cols_to_keep = ['Date'] + [col for col in prices.columns if '_pc' in col]
    prices_pattern = prices[cols_to_keep].dropna()
    prices_pattern = prices_pattern.drop(['Open_pc', 'High_pc', 'Low_pc', 'Close_pc'], axis=1)

    X = scaler.transform(prices_pattern.drop(['Date'], axis=1))
    X = X.reshape(X.shape[0], look_back, 4).astype(np.float32)
    return X

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        df = yf.download(ticker, period="9d")
        if df.empty:
            return jsonify({"error": "No data found"}), 404

        look_back = 7
        X = transform_data(look_back, df)
        preds = model.predict(X).tolist()

        results = []
        for p in preds:
            results.append({
                "down_prob": p[0] * 100,
                "up_prob": p[1] * 100
            })
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
