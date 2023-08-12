import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from PIL import Image

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Standardize the features for the selected features only
scaler = StandardScaler()
X_kbest = SelectKBest(score_func=f_classif, k=10).fit_transform(X, y)
X_scaled = scaler.fit_transform(X_kbest)

# Train an SVM classifier on the selected features
model = SVC(kernel='linear')
model.fit(X_scaled, y)

# Streamlit app
st.title("Breast Cancer Prediction")
st.sidebar.title("Feature Inputs")

# Function to get user input for predicting
def get_user_input(selected_feature_names):
    input_features = []
    for feature_name in selected_feature_names:
        value = st.sidebar.text_input(f"Enter {feature_name}:", "")
        try:
            # Attempt to convert the input value to float
            value = float(value)
        except ValueError:
            # Handle the case where the input cannot be converted to float
            st.warning(f"Invalid input for {feature_name}. Please enter a numeric value.")
            return None
        input_features.append(value)
    return np.array(input_features).reshape(1, -1)

# Get user input for the selected features
selected_feature_names = data.feature_names[SelectKBest(score_func=f_classif, k=10).fit(X, y).get_support()]
user_input = get_user_input(selected_feature_names)

# Display an image
image = Image.open("bc.jpg")
st.image(image, caption="Breast Cancer", use_column_width=True)

# Proceed if user input is valid
if user_input is not None:
    # Standardize the user input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Display prediction
    prediction_text = "Benign (0)" if prediction[0] == 0 else "Malignant (1)"
    st.success(f"Prediction: {prediction_text}")
