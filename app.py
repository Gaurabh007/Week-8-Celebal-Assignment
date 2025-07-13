import streamlit as st
from Loan_prediction import LoanAdvisor
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()


advisor = LoanAdvisor(groq_api_key=os.getenv("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"))
advisor.load_trained_model()

st.set_page_config(page_title="Loan Approval System", layout="centered")
st.title("🏦 Smart Loan Approval Predictor & Chatbot")

# Create tabs
tabs = st.tabs(["📋 Predict Loan Status", "🤖 Chat with AI"])

# ============================================= Tab 1: Loan Prediction ==========================================
with tabs[0]:
    st.header("Enter Applicant Details")

    # Form inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=1)
    loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 60])
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("Predict Approval"):
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }

        result, confidence = advisor.predict_applicant(input_data)
        st.success(f"✅ Prediction: **{result}** (Confidence: {confidence:.2%})")


# ==============================================================================================

st.title("🤖 AI Loan Chatbot")

client = advisor.groq_client

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if client is None:
    st.error("❌ AI Chatbot not available. Please set GROQ_API_KEY.")
else:
    st.markdown("Chat with our assistant to get help with your loan application.")

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    user_input = st.chat_input("Type your message here...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})


        extracted = advisor.extract_info(user_input)
        if extracted:
            st.session_state.user_data.update(extracted)

        with st.chat_message("assistant"):
            with st.spinner("AI is typing..."):
                reply = advisor.generate_ai_response(user_input, context=str(st.session_state.user_data))
                st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        required_fields = advisor.columns_used
        if all(field in st.session_state.user_data for field in required_fields):
            if st.button("📊 Predict Loan Approval"):
                try:
                    prediction, confidence = advisor.predict_applicant(st.session_state.user_data)
                    result = f"📢 Based on your info, the loan is likely to be **{prediction}** (Confidence: {confidence:.2%})"
                    st.chat_message("assistant").markdown(result)
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Sidebar control
    with st.sidebar:
        st.title("⚙️ Options")
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.user_data = {}
            st.rerun()

