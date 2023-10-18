# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from base64 import b64encode
from sklearn.preprocessing import StandardScaler

# Set Title
st.title('Client Satisfaction Prediction App')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file to make predictions", type=["csv"])


# Instantiate a StandardScaler
scaler = StandardScaler()

# Check if a CSV file is uploaded
if uploaded_file is not None:
    
    try:
        # Create a DataFrame with the data uploaded by the user
        data = pd.read_csv(uploaded_file)
        # Apply the scaler to the DataFrame
        processed_data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)

        
        # Display the test dataset to viwer
        st.subheader('Your Dataset')
        st.write(data)

        # Load models
        lr_model = joblib.load('LogisticRegressionModel.pkl')  
        gb_model = joblib.load('GradientBoostingModel.pkl')  
        nb_model = joblib.load('NaiveBayesModel.pkl')  
        rf_model = joblib.load('RandomForestModel.pkl') 
        svm_model = joblib.load('SupportVectorMachinesModel.pkl') 


        # Create a dropdown menu to select the model
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Gradient Boosting", "Naive Bayes","Random Forest","Support Vector Machines"])  # Add more models as needed

        # Create a button to make predictions
        predict_button = st.button("Predict")
        # Make predictionss
        if predict_button:
            if model_choice == "Logistic Regression":
                predictions = lr_model.predict(processed_data)  

            elif model_choice == "Gradient Boosting":
                predictions = gb_model.predict(processed_data)

            elif model_choice == "Naive Bayes":
                predictions = nb_model.predict(processed_data)

            elif model_choice == "Random Forest":
                predictions = rf_model.predict(processed_data)
            
            elif model_choice == "Support Vector Machines":
                predictions = svm_model.predict(processed_data)

            # Combine the predictions into a single DataFrame
            data['Predictions'] = predictions
        

            # Display the combined DataFrame with predictions
            st.subheader('Test Data with Predictions')
            st.write(data)

            # Add a download button for the DataFrame
            download_button = st.download_button(
                label="Download Predictions CSV",
                data=data.to_csv(index=False).encode('utf-8'),
                key="download_button",
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
