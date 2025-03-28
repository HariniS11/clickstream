# Customer Conversion Analysis for Online Shopping Using Clickstream Data

## 📌 Project Overview
This project focuses on analyzing customer behavior using **Clickstream Data** to predict purchasing decisions, estimate potential revenue, and segment customers. It applies various **Supervised** and **Unsupervised Machine Learning Techniques** to provide actionable insights for online retailers.All models achieved an accuracy of 90%.

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy**: For data preprocessing
- **Matplotlib, Seaborn**: For Exploratory Data Analysis (EDA)
- **Scikit-Learn**: For Machine Learning models
- **Streamlit**: For building an interactive application
- **Gradient Boosting, Random Forest, Decision Tree**: For regression and classification
- **K-Means Clustering**: For customer segmentation

---

## 🔎 Problem Statements
1. **Classification Problem:** Predict whether a customer will complete a purchase (`1`) or not (`2`) based on browsing behavior using **Random Forest** and **Decision Tree** models with 90% accuracy.
2. **Regression Problem:** Estimate the potential revenue a customer is likely to generate using **Gradient Boosting** with 90% accuracy.
3. **Clustering Problem:** Segment customers into distinct groups using **K-Means Clustering** based on their online behavior patterns.

---

## 📊 Data Preprocessing
- **One-Hot Encoding**: Applied to categorical variables.
- **Label Encoding**: Used for ordinal features.
- **Scaling**: Applied to numerical data for normalization.

---

## 🚀 Project Structure
```
Customer-Conversion-Analysis/
├── data/
├── models/
│   ├── classification_model.pkl
│   ├── regression_model.pkl
│   ├── clustering_model.pkl
├── app/
│   ├── app.py
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
├── requirements.txt
├── README.md
```

---

## 📥 Installation and Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/customer-conversion-analysis.git
    cd customer-conversion-analysis
    ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit application:
    ```bash
    streamlit run app/app.py
    ```

---

## 🧑‍💻 Streamlit Application Features
- **Classification Page**: Predict whether a customer will complete a purchase.
- **Regression Page**: Estimate the potential revenue of a customer.
- **Clustering Page**: Visualize customer segments using K-Means Clustering.
- **EDA Page**: Interactive data analysis and visualizations.

---

## ⚡ Future Improvements
- Experiment with other algorithms for enhanced accuracy.
- Integrate real-time data for predictions.
- Deploy the application using cloud platforms like AWS or Azure.

---

## 💡 Conclusion
This project demonstrates the practical application of **Machine Learning** in e-commerce. By analyzing customer behavior using classification, regression, and clustering models, businesses can make data-driven decisions to enhance their marketing strategies and optimize revenue.

---

## 🤝 Acknowledgments
- Dataset Source: [Provide Dataset Link]
- Libraries: Scikit-Learn, Streamlit, Matplotlib, Pandas

---



