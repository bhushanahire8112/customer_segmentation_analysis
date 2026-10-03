# Customer Segmentation

## 📌 Project Overview

This project uses **Machine Learning** to group customers into different segments based on their:

- Annual Income
- Spending Score

The project uses **K-Means Clustering** to find groups of similar customers.

A simple **Gradio UI** is also created where users can enter customer information and get the predicted customer segment.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Gradio

## 📂 Project Structure

```text
customer_segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── model/
│   ├── customer_scaler.pkl
│   └── customer_kmeans.pkl
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── app.py
└── README.md
```

## 🔍 Project Steps

### 1. Data Loading

The customer dataset is loaded using Pandas.

### 2. Data Understanding

The dataset is checked for:

- Shape
- Missing values
- Duplicate values
- Basic statistics

### 3. Feature Selection

The following features are used:

- Annual Income (k$)
- Spending Score (1-100)

### 4. Data Scaling

`StandardScaler` is used to scale the selected features before applying K-Means.

### 5. Finding the Number of Clusters

The **Elbow Method** is used to find a suitable number of clusters.

The selected number of clusters is **5**.

### 6. K-Means Clustering

K-Means is trained to group customers into 5 different clusters.

### 7. Cluster Analysis

The clusters are analyzed using their average income and spending score.

The clusters represent:

| Cluster | Customer Segment                 |
| ------- | -------------------------------- |
| 0       | Average Income, Average Spending |
| 1       | High Income, High Spending       |
| 2       | Low Income, High Spending        |
| 3       | High Income, Low Spending        |
| 4       | Low Income, Low Spending         |

### 8. Model Saving

The trained K-Means model and scaler are saved using Joblib.

### 9. Gradio UI

A simple Gradio interface is created where users enter:

- Annual Income
- Spending Score

The application then displays the customer's predicted cluster.

## 🚀 How to Run the Project

### Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib gradio
```

### Run the Application

Open the terminal inside the project folder and run:

```bash
python app.py
```

The Gradio application will open in the browser.

## 💡 Example

Input:

```text
Annual Income: 15
Spending Score: 80
```

Output:

```text
Cluster 2: Low Income, High Spending
```

## 🎯 Project Goal

The goal of this project is to understand **unsupervised learning and customer segmentation** using K-Means clustering and create a simple interface for making predictions.
