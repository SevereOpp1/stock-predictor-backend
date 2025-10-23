# Use a stable lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the files you need
COPY app.py /app/
COPY requirements.txt /app/
COPY stock_cnn.h5 /app/
COPY scaler.pkl /app/


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for the Flask app
EXPOSE 8080

# Run the Flask API when container starts
CMD ["python", "app.py"]
