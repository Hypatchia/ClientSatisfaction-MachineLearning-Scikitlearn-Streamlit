# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from base64 import b64encode
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time

# Read models from pkl files
# Load models

@st.cache_resource
def load_models():
    lr_model = joblib.load('streamlit/pkl_models/LogisticRegressionModel.pkl')  
    gb_model = joblib.load('streamlit/pkl_models/GradientBoostingModel.pkl')  
    nb_model = joblib.load('streamlit/pkl_models/NaiveBayesModel.pkl')  
    svm_model = joblib.load('streamlit/pkl_models/SupportVectorMachinesModel.pkl') 
    return lr_model, gb_model, nb_model, svm_model


@st.cache_data
def get_eda(data):
    
    # Display the test dataset to viwer
    st.subheader('Your Dataset')
    st.write(data)

    # Data Exploration Section
    st.header('Data Exploration')

    # Display basic statistics
    st.subheader('Summary Statistics')
    st.write(data.describe())

    st.subheader('Histograms')
    # Display histograms
    fig, ax = plt.subplots(figsize=(14, 14))
    data.hist(ax=ax)
    st.pyplot(fig)

   # Visualize the correlation matrix using a heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(14, 14))

    # Create a heatmap with seaborn
    sns.heatmap(data.corr(), annot=True, cmap='viridis', fmt='.2f', ax=ax)

    # Display the heatmap using streamlit
    st.pyplot(fig)

    return None


@st.cache_data
def get_predictions(data,model_choice):
# Create a DataFrame with the data uploaded by the user
    lr_model, gb_model, nb_model, svm_model = load_models()

    if model_choice == "Logistic Regression":
        predictions = lr_model.predict(data)  
        
    elif model_choice == "Gradient Boosting":
        predictions = gb_model.predict(data)

    elif model_choice == "Naive Bayes":
        predictions = nb_model.predict(data)

    elif model_choice == "Support Vector Machines":
        predictions = svm_model.predict(data)

    # Combine the predictions into a single DataFrame
    data['Predictions'] = predictions

    # Display the combined DataFrame with predictions
    st.subheader('Test Data with Predictions')
    st.write(data)

    # Add a download button for the DataFrame
    return data


@st.cache_resource
def load_data():
    data_path = 'streamlit/sample.csv'
    return pd.read_csv(data_path)

def main():
    # Set Title
    st.subheader('Customer Satisfaction: Predictive Analytics')
    # Display the test dataset to viwer
    st.write('INSTRUCTIONS:')
    st.write("1. Upload a CSV file to make predictions.")
    st.write("2. Get Instant Summary Statistics, Histograms, Correlation Matrix and Correlation Heatmap.")
    st.write("3. Then, Select a model of your choice.")
    st.write("4. Choose between Logistic Regression, Gradient Boosting, Naive Bayes, Support Vector Machines")
    st.write("5. Click the 'Predict' button.")
    st.write("6. View the test data with predictions.")
    st.write("7. Download the predictions CSV file.")

    

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file to make predictions", type=["csv"])
    
    st.subheader('For the purpose of this demo, data sample is pre-uploaded.')

    # Check if a CSV file is uploaded
    if uploaded_file is not None:
        try:
            data = load_data()
            get_eda(data)
          
        except:
            # Display an error if the file is not a CSV file
            st.error("Please upload a CSV file")
            raise Exception("File not CSV")

    else: 
            
            
            data= load_data()
       
            st.success('A Sample Dataset is Already Loaded')
            st.subheader('Get Exploratory Data Analysis')
            get_eda(data)
            st.subheader('Make Predictions')
         
        
            #get_eda(data)
                # Make predictions when the button is pressed
            
            model_choice = st.radio("Which Model would you like to use?",
                    options=['Logistic Regression', 'Gradient Boosting', 'Naive Bayes', 'Support Vector Machines'])
            if st.button("Predict"):
                st.write('Predictions are ready!')
                predictions = get_predictions(data, model_choice)
                st.download_button(
            label="Download Predictions CSV",
            data=predictions.to_csv(index=False).encode('utf-8'),
            key="download_button",
            file_name="predictions.csv",
            mime="text/csv",
            )
                
  
    

    
if __name__ == "__main__":
    main()

