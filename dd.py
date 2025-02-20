import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "dd.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for col in ["Sex", "BP", "Cholesterol", "Drug"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for decoding

# Define features and target variable
X = df.drop(columns=["Drug"])
y = df["Drug"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("💊 Drug Recommendation System")
st.write("Enter patient details to predict the recommended drug.")

# User inputs
age = st.slider("Age", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=30)

sex_options = label_encoders["Sex"].classes_
sex = st.selectbox("Sex", sex_options)
sex_encoded = label_encoders["Sex"].transform([sex])[0]

bp_options = label_encoders["BP"].classes_
bp = st.selectbox("Blood Pressure (BP)", bp_options)
bp_encoded = label_encoders["BP"].transform([bp])[0]

cholesterol_options = label_encoders["Cholesterol"].classes_
cholesterol = st.selectbox("Cholesterol", cholesterol_options)
cholesterol_encoded = label_encoders["Cholesterol"].transform([cholesterol])[0]

na_to_k = st.slider("Sodium-to-Potassium Ratio (Na_to_K)", float(df["Na_to_K"].min()), float(df["Na_to_K"].max()), value=15.0)

# Predict button
if st.button("Predict Drug"):
    input_data = pd.DataFrame([[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]], columns=X.columns)
    predicted_drug_encoded = model.predict(input_data)[0]
    
    # Decode the predicted drug
    predicted_drug = label_encoders["Drug"].inverse_transform([predicted_drug_encoded])[0]
    
    st.success(f"🏥 Recommended Drug: **{predicted_drug}**")

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"📊 **Model Accuracy:** {accuracy:.2f}")

# Decision Tree Visualization
st.subheader("🌳 Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(model, feature_names=X.columns, class_names=label_encoders["Drug"].classes_, filled=True, ax=ax)
st.pyplot(fig)
