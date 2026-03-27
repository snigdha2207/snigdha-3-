
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

st.title("Petite Fashion Analytics Dashboard")

data = pd.read_csv("petite_dataset.csv")

st.subheader("Dataset Preview")
st.write(data.head())

X = data.drop("purchase_intent", axis=1)
X = pd.get_dummies(X)
y = data["purchase_intent"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Model Performance")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))

st.subheader("Feature Importance")
importances = model.feature_importances_
feat_df = pd.DataFrame({"feature":X.columns, "importance":importances})
st.write(feat_df.sort_values(by="importance", ascending=False))
