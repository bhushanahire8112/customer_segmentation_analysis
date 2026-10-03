import joblib
import pandas as pd

# Load saved model and scaler
scaler = joblib.load('customer_segmentation/model/customer_scaler.pkl')
kmeans = joblib.load('customer_segmentation/model/customer_kmeans.pkl')

# Take input from user
income = float(input("Enter Annual Income (k$): "))
spending = float(input("Enter Spending Score (1-100): "))

# Create input data with feature names
customer = pd.DataFrame(
    [[income, spending]],
    columns=["Annual Income (k$)", "Spending Score (1-100)"]
)

# Scale the input
customer_scaled = scaler.transform(customer)

# Predict cluster
cluster = kmeans.predict(customer_scaled)

print("Customer belongs to Cluster:", cluster[0])