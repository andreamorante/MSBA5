# Load Packages 
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data Source: social media usage
s = pd.read_csv("social_media_usage.csv")

# Specify relevant columns to keep
subset_df = s[['income', 'educ2', 'age', 'par', 'marital', 'web1h', 'gender']].copy()

# Rename columns 
subset_df.rename(columns={
        'educ2': 'education',
        'par': 'parent',
        'marital': 'married'
            }, inplace=True)

# Define a function to clean social media usage column
def clean_sm(x): 
    return np.where(x == 1, 1, 0)

# Create dataframe ss and target column
ss = pd.DataFrame(subset_df).copy()
ss['sm_li'] = clean_sm(ss['web1h']).copy()  # New column
ss.drop('web1h', axis=1, inplace=True)

# Process features as valid values, others set to NaN and others as binary
ss['income'] = ss['income'].apply(lambda x: x if 1 <= x <= 9 else 0)
ss['education'] = ss['education'].apply(lambda x: x if 1 <= x <= 8 else 0)
ss['parent'] = ss['parent'].apply(lambda x: 1 if x == 1 else 0)
ss['married'] = ss['married'].apply(lambda x: 1 if x == 1 else 0)
ss['gender'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
ss['age'] = ss['age'].apply(lambda x: x if x <= 98 else 0)

# Define features and target ## Split data into training and test set
X = ss.drop(columns=['sm_li'])  # Drop the target column to keep features
y = ss['sm_li']     

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=987)

# Train Logistic Regression Model
ss_model = LogisticRegression(class_weight='balanced')
ss_model.fit(X_train, y_train)

# Define feature labels
income_labels = [
    "Less than $10,000", "$10,000-$19,999", "$20,000-$29,999",
    "$30,000-$39,999", "$40,000-$49,999", "$50,000-$74,999",
    "$75,000-$99,999", "$100,000-$149,999", "$150,000 or more"
]
education_labels = [
    "Less than high school", "High school incomplete", "High school graduate",
    "Some college, no degree", "Associate degree", "Bachelor’s degree",
    "Master’s degree", "Doctorate or professional degree"
]

# Function to preprocess inputs
def preprocess_input(income, education, age, parent, married, gender):
    # Ensure the inputs are within valid ranges, otherwise set to NaN
    income = income if 1 <= income <= 9 else np.nan
    education = education if 1 <= education <= 8 else np.nan
    # Age is binned into categories
    age_bin = (
        1 if age <= 18 else
        2 if age <= 35 else
        3 if age <= 55 else
        4 if age <= 75 else
        5 if age <= 98 else np.nan
    )
    # Returning the processed features
    return [
        income, education, age_bin,
        1 if parent == "Yes" else 0,
        1 if married == "Yes" else 0,
        1 if gender == "Female" else 0
    ]

# Streamlit App
st.title("LinkedIn User Prediction App")
st.write("Enter your details to predict LinkedIn usage and probability.")

# User inputs
income = st.selectbox("Income Range", range(1, 10), format_func=lambda x: income_labels[x - 1])
education = st.selectbox("Education Level", range(1, 9), format_func=lambda x: education_labels[x - 1])
age = st.number_input("Age (years)", min_value=0, max_value=98, step=1)
parent = st.radio("Are you a parent?", ["Yes", "No"])
married = st.radio("Are you married?", ["Yes", "No"])
gender = st.radio("Gender", ["Female", "Male"])


# Predict button
if st.button("Predict"):
    # Preprocess the input features
    features = preprocess_input(income, education, age, parent, married, gender)

    if np.nan in features:
        st.error("Some inputs are invalid. Please check and try again.")
    else:
        # Convert features to a DataFrame with the same columns as the training data
        features_df = pd.DataFrame([features], columns=X_train.columns)
        
        # Make prediction and calculate probabilities
        prediction = ss_model.predict(features_df)[0]
        probabilities = ss_model.predict_proba(features_df)[0]

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**LinkedIn User?** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability of being a LinkedIn user:** {probabilities[1]:.2%}")

        # Prediction confidence (Bar chart)
        st.subheader("Prediction Confidence")
        confidence_df = pd.DataFrame({
            "Category": ["LinkedIn User", "Not LinkedIn User"],
            "Probability": [probabilities[1], probabilities[0]]
        })
        st.bar_chart(confidence_df.set_index("Category"))
