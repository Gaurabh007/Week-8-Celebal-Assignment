import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import re
import pickle
import os
from groq import Groq

class LoanAdvisor:
    def __init__(self, groq_api_key=None):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.columns_used = []
        self.groq_client = None

        api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                print("✅ AI Chatbot initialized successfully!")
            except Exception as e:
                print(f"❌ Failed to initialize AI Chatbot: {e}")
        else:
            print("⚠️ No Groq API key provided. AI Chatbot features will be disabled.")

    def load_data(self, path):
        df = pd.read_csv(path)
        print("Data loaded. Shape:", df.shape)
        return df

    def clean_data(self, df):
        df = df.copy()
        df.fillna(method='ffill', inplace=True)
        df['Dependents'] = df['Dependents'].replace('3+', '3').astype(float)
        return df

    def engineer_features(self, df):
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomeRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        return df

    def encode_and_scale(self, df, training=True):
        df = df.copy()
        cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        for col in cols:
            if training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                df[col] = le.transform(df[col].astype(str))

        self.columns_used = [c for c in df.columns if c not in ['Loan_ID', 'Loan_Status']]
        X = df[self.columns_used]
        y = df['Loan_Status'].map({'Y': 1, 'N': 0}).values if 'Loan_Status' in df.columns else None

        if training:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_save(self, X, y, model_path="loan_model.h5", components_path="preprocessing.pkl"):
        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)
        model = self.build_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        print("\nClassification Report:\n", classification_report(y_val, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
        print("ROC AUC:", roc_auc_score(y_val, y_pred))

        model.save(model_path)
        with open(components_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'encoders': self.encoders,
                'columns_used': self.columns_used
            }, f)
        print(f"Model saved to {model_path} and preprocessing components to {components_path}")

    def load_trained_model(self, model_path="models/loan_model.h5", components_path="models/preprocessing.pkl"):
        self.model = load_model(model_path)
        with open(components_path, 'rb') as f:
            components = pickle.load(f)
            self.scaler = components['scaler']
            self.encoders = components['encoders']
            self.columns_used = components['columns_used']
        print("Model and preprocessing components loaded successfully.")

    def predict_applicant(self, info):
        df = pd.DataFrame([info])
        df = self.engineer_features(df)
        for col in self.encoders:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        for col in self.columns_used:
            if col not in df.columns:
                df[col] = 0
        X = df[self.columns_used]
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict(X_scaled)[0][0]
        return ("Approved" if prob > 0.5 else "Rejected"), float(prob)

    def generate_ai_response(self, user_message, context=""):
        if not self.groq_client:
            return "❌ AI Chatbot not initialized. Please provide a valid API key."

        try:
            system_prompt = f"""You are a helpful loan advisor assistant. You help users understand loan approval processes and gather necessary information for loan applications.

Context about the loan application system:
- The system predicts loan approval based on factors like income, credit history, education, employment status, etc.
- Required information: Gender, Marital Status, Dependents, Education, Employment Status, Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area

Current context: {context}

Please provide helpful, accurate, and friendly responses about loan applications. If asked about specific loan details, guide the user through the application process step by step."""

            completion = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            return completion.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def extract_info(self, message):
        details = {}

        income_match = re.search(r"income.*?(\d+)", message.lower())
        if income_match:
            details['ApplicantIncome'] = int(income_match.group(1))

        loan_match = re.search(r"loan.*?amount.*?(\d+)", message.lower())
        if loan_match:
            details['LoanAmount'] = int(loan_match.group(1))

        if 'male' in message.lower() and 'female' not in message.lower():
            details['Gender'] = 'Male'
        elif 'female' in message.lower():
            details['Gender'] = 'Female'

        if 'married' in message.lower():
            details['Married'] = 'Yes'
        elif 'single' in message.lower() or 'unmarried' in message.lower():
            details['Married'] = 'No'

        if 'graduate' in message.lower():
            details['Education'] = 'Graduate'
        elif 'not graduate' in message.lower():
            details['Education'] = 'Not Graduate'

        return details

if __name__ == "__main__":
    advisor = LoanAdvisor()

    # Step 1: Load and prepare dataset
    dataset_path = "data/Training Dataset.csv"  # Replace with your actual dataset path
    df = advisor.load_data(dataset_path)
    df = advisor.clean_data(df)
    df = advisor.engineer_features(df)
    X, y = advisor.encode_and_scale(df)

    # Step 2: Train and save model
    advisor.train_and_save(X, y)

    print("\n✅ Model training complete and saved.")
    print("📦 You can now use the Streamlit app or other frontend to interact with it.")
