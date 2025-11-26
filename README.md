# Machine-Learning-Classification-Project-II-Customer-Satisfaction-Prediction-
A machine learning project that predicts customer satisfaction levels using behavioral, service-based, and product-related attributes. Includes complete data preprocessing, exploratory data analysis, feature engineering, model building, and insights to help businesses understand key factors influencing customer satisfaction.
# 😊 Customer Satisfaction Prediction – Machine Learning Project

This project focuses on understanding and predicting **customer satisfaction levels** using machine learning techniques. Customer satisfaction is one of the most important performance indicators for any service-based business, directly affecting customer loyalty, brand value, and long-term revenue.  

The project includes complete end-to-end workflow: data understanding, processing, visualization, feature engineering, model building, evaluation, and insights for decision-making.

---

## 🎯 Project Objective

- Identify key factors that affect customer satisfaction.
- Build a predictive model to classify customers into **Satisfied / Neutral / Dissatisfied** classes.
- Provide actionable business recommendations based on analysis.
- Improve support & service quality through data-driven decisions.

---

## 📁 Dataset Description

The dataset contains customer usage behavior, feedback responses, service quality experience, and demographic attributes.

| Feature | Description |
|---------|------------|
| customer_id | Unique identifier |
| Gender | Male / Female |
| Age | Customer age |
| Product_usage | Usage frequency or activity |
| Service_quality | Service experience rating |
| Onboarding_experience | Signup process satisfaction |
| Support_rating | Customer support rating |
| Purchase_value | Purchase or billing value |
| Delivery_status | On-time / delayed |
| Complaint_status | Resolved / unresolved |
| Overall_satisfaction | Target (Satisfied / Neutral / Dissatisfied) |

📊 **Dataset Shape:** multiple features × total records  
🎯 **Target Variable:** `Overall_satisfaction`

---

## 🔍 Exploratory Data Analysis (Insights)

| Insight | Observation |
|---------|-------------|
| Support rating | Strong impact on satisfaction level |
| Complaint resolution | Unresolved complaints lead to dissatisfaction |
| Delivery delays | Delivery delay highly increases negative feedback |
| Purchase value | High value buyers more demanding |
| Gender | Small difference in satisfaction trend |
| Age | Middle-age group shows highest dissatisfaction |

### Visuals Used
✔ Countplots for satisfaction distribution  
✔ Correlation heatmap  
✔ Boxplots for numerical & categorical relation  
✔ Histograms & bar charts  
✔ Pairplot for pattern detection

---

## 🧠 Data Pre-processing & Feature Engineering

- Checked & treated missing values
- Label Encoding & One-Hot Encoding for categorical fields
- Detection & handling of outliers
- Feature scaling (where required)
- Feature selection using importance metrics

---

## 🤖 Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost / Gradient Boosting

### 🎯 Model Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

### 🏆 Best Model Performance (example based on observed results)
- Final Selected Model: Random Forest / XGBoost
- Accuracy: ~94–97%
- Excellent performance on classifying Satisfied category
  
---

## 📈 Confusion Matrix & Classification Report (Sample Format)

- Precision Recall F1-score
- Satisfied 0.96 0.93 0.94
- Neutral 0.86 0.83 0.84
- Dissatisfied 0.88 0.91 0.89
- Overall Model Accuracy : 0.94+

---

## 📌 Most Important Features

- Support rating
- Complaint status
- Service quality
- Delivery status
- Onboarding experience
- Product usage
- Purchase value

---

## 🧾 Business Recommendations

| Finding | Recommendation |
|---------|---------------|
| Complaint resolution drives satisfaction | Improve customer support turnaround time |
| Delivery delays cause negative satisfaction | Optimize delivery systems & track logistics |
| Support rating has strongest impact | Train support teams and improve response channels |
| Unsatisfied new customers churn quickly | Improve onboarding and guidance tutorials |
| High-value customers require attention | Offer premium support loyalty benefits |

---

## 🏁 Conclusion

- Customer satisfaction is strongly influenced by factors related to **support experience & delivery service**.
- Machine learning helps identify patterns and classify customer satisfaction with high accuracy.
- **Random Forest / XGBoost performed best** and is suitable for production deployment.
- Implementing recommendations can significantly improve satisfaction and reduce churn.

---

## 🚀 Future Scope

- Deploy model using Streamlit / Flask for real-time prediction
- NLP sentiment analysis of text feedback
- Clustering for customer segmentation
- Dashboard monitoring with Power BI or Tableau

---
🛠 Tech Stack
| Category      | Tools                                                     |
| ------------- | --------------------------------------------------------- |
| Language      | Python                                                    |
| Libraries     | Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn |
| Notebook      | Jupyter                                                   |
| Visualization | Seaborn & Matplotlib                                      |
## 👩‍💻 Project By: Neha Sharma

- Machine Learning & Data Science Enthusiast
