import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import xgboost
import joblib

# read data
df = pd.read_csv("train_ctrUa4K.csv")

# layout
st.set_page_config(page_title="Loan Detection", layout="wide")

pages = st.sidebar.radio("Pages", ["Data", "Predict state your Loan"])

if pages == "Data":
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2.3, 0.1, 1.3, 0.1)
    )

    with row0_1:
        st.subheader("Data Describtion : ")
        st.write(
            """
                    | Attribute | Description |
                    |----------|----------|
                    |Loan_ID	|Unique Load ID
                    |Gender	|Male / Female
                    |Married	|Yes / No
                    |Dependents |Number of dependents
                    |Education	|Applicant Education (Graduate / Not Graduate)
                    |Self_Employed	|Self Employed (Yes / No)
                    |ApplicantIncome  |Applicant Income
                    |CoapplicantIncome	|Co applicant Income
                    |LoanAmount	|Loan Amount in thousands
                    |Loan_Amount_Term	|Term of loan in months
                    |Credit_History	|Credit History meets guidelines
                    |Property_Area	|Urban / semi Urban / rural
                    |Loan_Status |(Target) Loan approved (Y/N)
                                                            """
        )
    with row0_2:
        st.text("")
        st.subheader(
            "Linkedin : App by [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/) "
        )
        st.subheader(
            "Github : App by [Ahmed Ramadan](https://github.com/AhmedRamadan74/Loan-Detection)"
        )

    st.subheader("Display first 10 rows of data : ")
    st.dataframe(df.head(10))


if pages == "Predict state your Loan":
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
        (0.1, 2.3, 0.1, 1.3, 0.1)
    )

    with row0_1:
        st.title("Loan project Desctiption")
        st.markdown(
            """ <h6>
                        Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the the loan eligibility process (real time) based on customer detail provided while filling online application form. </center> </h6> """,
            unsafe_allow_html=True,
        )
    with row0_2:
        st.text("")
        st.subheader(
            "Linkedin : App by [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-18b873230/) "
        )
        st.subheader(
            "Github : App by [Ahmed Ramadan](https://github.com/AhmedRamadan74/Loan-Detection)"
        )
    st.markdown("*" * 50)
    st.header("The aim of the project : ")
    st.write(
        "Build Machine learning model to help company to automate the loan eligibility process (real time) based on customer detail provided while filling online application form"
    )
    model = joblib.load("model.pkl")  # load model
    inputs = joblib.load("input.pkl")  # load input

    def Make_Prediction(
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area,
    ):
        df_pred = pd.DataFrame(columns=inputs)
        df_pred.at[0, "Gender"] = Gender
        df_pred.at[0, "Married"] = Married
        if Dependents >= 3:
            df_pred.at[0, "Dependents"] = 3
        else:
            df_pred.at[0, "Dependents"] = Dependents
        df_pred.at[0, "Education"] = Education
        df_pred.at[0, "Self_Employed"] = Self_Employed
        df_pred.at[0, "ApplicantIncome"] = ApplicantIncome
        df_pred.at[0, "CoapplicantIncome"] = CoapplicantIncome
        df_pred.at[0, "LoanAmount"] = LoanAmount / 1000
        df_pred.at[0, "Loan_Amount_Term"] = Loan_Amount_Term
        df_pred.at[0, "Credit_History"] = Credit_History
        df_pred.at[0, "Property_Area"] = Property_Area
        # prediction output
        result = model.predict_proba(df_pred)[:, 1][0]
        if result >= 0.858:  # thershold
            return "Your loan approve"
        else:
            return "Sorry your loan rejected"

    st.subheader("Enter customer detail : ")
    Gender = st.selectbox("Gender of customer :", ["Male", "Female"])
    Married = st.selectbox("Customer is married (Y/N):", ["Yes", "No"])
    Dependents = st.number_input(
        "Number of customer's dependents : ", min_value=0, step=1
    )
    Education = st.selectbox(
        " Education customer state :", ["Graduate", "Not Graduate"]
    )
    Self_Employed = st.selectbox(" Customer is Self Employed (Y/N) :", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income : ")
    CoapplicantIncome = st.number_input(
        "Co applicant Income  if you don't have Co applicant enter 0: "
    )
    LoanAmount = st.number_input("Loan Amount : ")
    Loan_Amount_Term = st.number_input("Term of loan in months : ")
    Credit_History = st.selectbox("Credit History meets guidelines:", [0, 1])
    Property_Area = st.selectbox(
        " Property Area of Customer :", ["Urban", "semi Urban", "rural"]
    )
    # show the result
    btn = st.button("Predict")
    if btn:
        st.write(
            Make_Prediction(
                Gender,
                Married,
                Dependents,
                Education,
                Self_Employed,
                ApplicantIncome,
                CoapplicantIncome,
                LoanAmount,
                Loan_Amount_Term,
                Credit_History,
                Property_Area,
            )
        )
