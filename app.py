import streamlit as st
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler

# Set up page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load the saved model
model = pkl.load(open('trained_model.sav', 'rb'))

# Prediction function
def predict_disease(x):
    return model.predict([x])

# Preprocessing user input
def preprocess(age, sex, cholesterol, heartrate, diabetes, familyhistory, smoking, obesity,
               exercisehoursperweek, diet, previousheartproblems, medicationuse,
               sedentaryhoursperday, bmi, triglycerides, physicalactivitydaysperweek,
               sleephoursperday, BP_Systolic, BP_Diastolic):
    
    # Convert categorical values to numerical
    sex = 1 if sex == "male" else 0
    diabetes = 1 if diabetes == "I've diabetes" else 0
    familyhistory = 1 if familyhistory == "I've heart disease family history" else 0
    smoking = 1 if smoking == "I smoke" else 0
    obesity = 1 if obesity == "I have obesity" else 0

    diet_mapping = {"Healthy": 1, "Average ": 0, "Unhealthy": 2}
    diet = diet_mapping.get(diet, 0)

    previousheartproblems = 1 if previousheartproblems == "Yes" else 0
    medicationuse = 1 if medicationuse == "yes" else 0

    # Create input array
    x = np.array([
        age, sex, cholesterol, heartrate, diabetes, familyhistory, smoking, obesity,
        exercisehoursperweek, diet, previousheartproblems, medicationuse,
        sedentaryhoursperday, bmi, triglycerides, physicalactivitydaysperweek,
        sleephoursperday, BP_Systolic, BP_Diastolic
    ])

    # Standardize input
    scaler = StandardScaler()
    x = scaler.fit_transform(x.reshape(1, -1))
    return x[0]

# Frontend Design
html_temp = '''
<div style="text-align:center;">
    <h1 style="color:black;">Heart Attack Risk Prediction</h1>
</div>
'''

st.markdown(html_temp, unsafe_allow_html=True)
st.image("logo.png", width=500)

# Input fields
age = st.selectbox("Age", range(20, 90, 1))
sex = st.radio("Select Gender:", ('male', 'female'))
heartrate = st.selectbox("Heart Rate BPM", range(40, 120, 1))
diabetes = st.radio("Diabetes state:", ("I've diabetes", "I don't have diabetes"))
familyhistory = st.radio("Family history:", ("I've heart disease family history", "I don't have heart disease family history"))
smoking = st.radio("Do you smoke:", ('I smoke', 'I donot smoke'))
obesity = st.radio("Do you have obesity:", ('I have obesity', 'I donot have obesity'))
diet = st.selectbox("Diet Type", ("Healthy", "Average ", "Unhealthy"))
previousheartproblems = st.radio("Previous Heart Problems:", ("Yes", "No"))
medicationuse = st.radio("Do you take medication:", ('yes', 'No'))
bmi = st.number_input("Body Mass Index (BMI)")
cholesterol = st.selectbox("Cholesterol (mg/dL)", range(160, 400, 1))
sedentaryhoursperday = st.selectbox("Sedentary Hours Per Day", range(0, 12, 1))
sleephoursperday = st.selectbox("Sleep Hours Per Day", range(1, 11, 1))
exercisehoursperweek = st.selectbox("Exercise Hours Per Week", range(0, 24, 1))
physicalactivitydaysperweek = st.selectbox("Physical Activity Days Per Week", range(0, 7, 1))
triglycerides = st.number_input("Triglycerides value")
BP_Systolic = st.number_input("Blood Pressure (Systolic)")
BP_Diastolic = st.number_input("Blood Pressure (Diastolic)")

# Prediction
if st.button("Predict"):
    user_processed_input = preprocess(
        age, sex, cholesterol, heartrate, diabetes, familyhistory, smoking, obesity,
        exercisehoursperweek, diet, previousheartproblems, medicationuse,
        sedentaryhoursperday, bmi, triglycerides, physicalactivitydaysperweek,
        sleephoursperday, BP_Systolic, BP_Diastolic
    )
    pred = predict_disease(user_processed_input)

    if pred[0] == 0:
        st.success('You have a lower risk of getting a heart attack!')
    else:
        st.error('Warning! You have a high risk of getting a heart attack!')

# Sidebar
st.sidebar.subheader("About App")
st.sidebar.info("This web app helps you find out whether you are at risk of a heart attack.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check your risk level.")
