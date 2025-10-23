# stock-predictor-backend
Serverless backend powering a Streamlit-based stock predictor — built with AWS Fargate and Python

The backend receives a stock ticker from the Streamlit UI and returns a prediction using a trained deep learning model (CNN).

Overview

Flask REST API serving stock predictions

Dockerized and deployed on AWS Fargate (serverless compute)

Communicates with the Streamlit frontend via an HTTP endpoint

Files

app.py: main backend entry point

model.h5: trained model file

requirements.txt: Python dependencies

Usage

The backend expects a POST request with JSON data:

{ "ticker": "AAPL" }


It responds with a prediction result:

{"predictions":[{"down_prob":28.268155455589294,"up_prob":71.73184156417847}]}

down_prob: probability stock price dips below yesterday's close
up_prob: probability stock price rises above yesterday's close
