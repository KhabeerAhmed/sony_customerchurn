# Import the necessary libraries
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(
    page_title="Customer Churn Prediction App",  # Title of the app
    layout="wide",  # Use wide layout
    initial_sidebar_state="expanded",  # Sidebar starts expanded
)

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of app.py
model_path = os.path.join(BASE_DIR, '../models/best_model.pkl')
model = joblib.load(model_path)

# Define required columns
required_columns = [
    'international plan', 'voice mail plan', 'number vmail messages',
    'total day minutes', 'total day charge',
    'total eve minutes', 'total eve charge',
    'total night minutes', 'total night charge',
    'total intl minutes', 'total intl charge',
    'customer service calls'
]

# Sidebar Navigation
st.sidebar.title("🗂 Navigation")
st.sidebar.markdown("### 📌 Options")
st.sidebar.markdown("- **🏠 Home**")
st.sidebar.markdown("- **📂 Upload Data**")
st.sidebar.markdown("- **📊 Predictions**")
st.sidebar.markdown("---")  # Horizontal divider

# Add a link to the default test data in the sidebar
st.sidebar.markdown("### 📝 Use Default Test Data")
st.sidebar.markdown(
    "[📥 Download Test Data](https://raw.githubusercontent.com/KhabeerAhmed/sony_customerchurn/refs/heads/main/data/processed/test_data.csv)"
)


# App Title and Description
st.title("📊 Customer Churn Prediction")
st.markdown("""
This application is created to predict customer churn at a telecom company, the dataset has been provided by **Sony Research**. The purpose of this app is to predict customer churn based on historical data and evaluate the performance of the predictive model. More exploratory data analysis and modeling can be found on my [Github](https://github.com/KhabeerAhmed/sony_customerchurn/tree/main) profile under 'notebooks' or 'reports'.

For more details about the assignment, visit the link below:
- [Customer Churn Prediction Assignment on Stratascratch](https://platform.stratascratch.com/data-projects/customer-churn-prediction)
""")

with st.expander("❓ How to Use This App"):
    st.write("1. Upload a CSV file with customer data.")
    st.write("2. Download the test data from the sidebar to obtain consistent results.")
    st.write("3. Ensure the file contains the mandatory columns.")
    st.write("4. Review predictions and download results.")

# Display the mandatory columns message
st.markdown("**Mandatory Columns Required for Prediction:**")
st.markdown(f"""```{', '.join(required_columns)}```""")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a CSV file:", type=['csv'])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
else:
    st.warning("📂 Please upload a file to continue.")

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Use columns to display input data and predictions side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Input Data")
        st.dataframe(input_data)  # Display input data in the first column

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        st.error(f"❌ The uploaded file is missing the following mandatory columns: {', '.join(missing_columns)}")
    else:
        filtered_data = input_data[required_columns]
        filtered_data['international plan'] = filtered_data['international plan'].astype('category')
        filtered_data['voice mail plan'] = filtered_data['voice mail plan'].astype('category')

        # Check for missing values
        if filtered_data.isnull().values.any():
            st.error("❌ The data contains missing values. Please clean and re-upload.")
        else:
            predictions = model.predict(filtered_data)
            probabilities = model.predict_proba(filtered_data)[:, 1]
            filtered_data['Prediction'] = predictions
            filtered_data['Churn Probability'] = probabilities

            with col2:
                st.subheader("📊 Predictions")
                st.dataframe(filtered_data)  # Display predictions in the second column

            # Display Metrics
            st.subheader("📊 Model Metrics")
            col3, col4 = st.columns(2)

            # Add a pie chart for visualization
            churn_counts = filtered_data['Prediction'].value_counts()
            fig = px.pie(
                names=["No Churn", "Churn"],
                values=churn_counts,
                title="Churn Prediction Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig.update_traces(hoverinfo="label+percent", textinfo="value")
            st.plotly_chart(fig)

            with col3:
                # Use the churn column from the uploaded data for true labels
                true_labels = input_data['churn']

                # Churn Rate
                churn_rate = (true_labels.sum() / len(true_labels)) * 100

                # Retention Rate
                retention_rate = 100 - churn_rate

                # Display the metrics using st.write
                st.write("**Rate Metrics**")
                st.write(f"- **Churn Rate:** {churn_rate:.2f}%")
                st.write(f"- **Retention Rate:** {retention_rate:.2f}%")

            with col4:
                # Classification Report as a DataFrame
                st.write("**Classification Metrics**")
                
                # Generate classification report as dictionary using true labels
                report = classification_report(
                    true_labels,
                    filtered_data['Prediction'],
                    output_dict=True
                )
                
                # Convert to DataFrame
                report_df = pd.DataFrame(report).transpose()
                
                # Display the DataFrame
                st.dataframe(report_df.style.format(precision=2))

            # Confusion Matrix
            with st.expander("Confusion Matrix"):
                st.write("Confusion Matrix")
                cm = confusion_matrix(
                    true_labels,
                    filtered_data['Prediction']
                )
                st.dataframe(
                    pd.DataFrame(cm, columns=["Predicted No Churn", "Predicted Churn"], 
                                index=["Actual No Churn", "Actual Churn"]),
                    use_container_width=True
                )

            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {len(filtered_data)} Predictions as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

st.markdown("""
<hr>
<p style='text-align: center;'>
    Developed by <a href="https://github.com/KhabeerAhmed" target="_blank">Khabeer Ahmed</a> | 
    <a href="https://www.linkedin.com/in/khabeerahmed" target="_blank">LinkedIn</a> © 2024
</p>
""", unsafe_allow_html=True)