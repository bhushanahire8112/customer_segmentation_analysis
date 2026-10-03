import gradio as gr
import joblib
import pandas as pd

# Load model and scaler
scaler = joblib.load("customer_segmentation/model/customer_scaler.pkl")
kmeans = joblib.load("customer_segmentation/model/customer_kmeans.pkl")


# Prediction function
def predict_cluster(income, spending):

    # Create DataFrame
    customer = pd.DataFrame(
        [[income, spending]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )

    # Scale input
    customer_scaled = scaler.transform(customer)

    # Predict cluster
    cluster = kmeans.predict(customer_scaled)[0]

    # Cluster descriptions
    descriptions = {
        0: "Average Income, Average Spending",
        1: "High Income, High Spending",
        2: "Low Income, High Spending",
        3: "High Income, Low Spending",
        4: "Low Income, Low Spending"
    }

    return f"Cluster {cluster}: {descriptions[cluster]}"


# Create Gradio UI
app = gr.Interface(
    fn=predict_cluster,
    inputs=[
        gr.Number(label="Annual Income (k$)"),
        gr.Number(label="Spending Score (1-100)")
    ],
    outputs=gr.Textbox(label="Customer Segment"),
    title="Customer Segmentation",
    description="Enter customer income and spending score to find the customer segment."
)

# Run app
app.launch()