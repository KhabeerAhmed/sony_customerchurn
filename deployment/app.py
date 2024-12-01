import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the model
model = joblib.load('../models/best_model.pkl')

# Define required columns
required_columns = [
    'international plan', 'voice mail plan', 'number vmail messages',
    'total day minutes', 'total day charge',
    'total eve minutes', 'total eve charge',
    'total night minutes', 'total night charge',
    'total intl minutes', 'total intl charge',
    'customer service calls'
]

# App Title and Description
st.title("📊 Customer Churn Prediction App")
st.markdown("Upload customer data to predict the likelihood of **churning**.")

# Display the mandatory columns message
st.markdown("**Mandatory Columns Required for Prediction:**")
st.markdown(f"""```{', '.join(required_columns)}```""")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a CSV file:", type=['csv'])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    st.markdown("### 📋 Input Data")
    st.dataframe(input_data.head())

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

            st.markdown("### 📊 Predictions")
            st.dataframe(filtered_data)

            # Add a pie chart for visualization
            churn_counts = filtered_data['Prediction'].value_counts()
            fig = px.pie(
                names=["No Churn", "Churn"],
                values=churn_counts,
                title="Churn Prediction Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig)

            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )
