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

st.sidebar.title("🗂 Navigation")
st.sidebar.markdown("### 📌 Options")
st.sidebar.markdown("- **🏠 Home**")
st.sidebar.markdown("- **📂 Upload Data**")
st.sidebar.markdown("- **📊 Predictions**")
st.sidebar.markdown("---")  # Horizontal divider

# App Title and Description
st.title("📊 Customer Churn Prediction App")
st.markdown("Upload customer data to predict the likelihood of **churning**.")

with st.expander("❓ How to Use This App"):
    st.write("1. Upload a CSV file with customer data.")
    st.write("2. Ensure the file contains the mandatory columns.")
    st.write("3. Review predictions and download results.")

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
    Developed by [Khabeer Ahmed](https://github.com/KhabeerAhmed) | (www.linkedin.com/in/khabeerahmed) © 2024
</p>
""", unsafe_allow_html=True)
