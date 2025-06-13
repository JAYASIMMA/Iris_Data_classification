import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix, classification_report

st.set_page_config(layout="wide")
st.title("🌸 Iris Species Classification App")

# Load and prepare the data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].map(dict(enumerate(iris.target_names)))

# Sidebar: Choose view
option = st.sidebar.selectbox("Choose Section", ["Dataset Overview", "Visualizations", "Model Training", "Make Predictions"])

# Sidebar for model training
st.sidebar.markdown("---")
st.sidebar.header("Prediction Input")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

le = LabelEncoder()
y_test_enc = le.fit_transform(y_test)
y_pred_enc = le.transform(y_pred)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test_enc, y_pred_enc)
mae = mean_absolute_error(y_test_enc, y_pred_enc)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Sections
if option == "Dataset Overview":
    st.subheader("Iris Dataset")
    st.dataframe(df.head())
    st.markdown(f"**Shape:** {df.shape}")
    st.markdown("**Target Names:**")
    st.write(iris.target_names)

elif option == "Visualizations":
    st.subheader("Data Visualizations")

    st.markdown("### Pairplot")
    fig1 = sns.pairplot(df, hue="species")
    st.pyplot(fig1)

    st.markdown("### Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Boxplot")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df.iloc[:, :-1], orient="h", ax=ax3)
    st.pyplot(fig3)

    st.markdown("### Violin Plot of Petal Length by Species")
    fig4, ax4 = plt.subplots()
    sns.violinplot(x="species", y="petal length (cm)", data=df, ax=ax4)
    st.pyplot(fig4)

    st.markdown("### Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig5, ax5 = plt.subplots()
    importances.plot(kind='barh', ax=ax5)
    st.pyplot(fig5)

    st.markdown("### Confusion Matrix")
    fig6, ax6 = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax6)
    st.pyplot(fig6)

    st.markdown("### Histogram of Petal Length")
    fig7, ax7 = plt.subplots()
    sns.histplot(df['petal length (cm)'], kde=True, bins=20, ax=ax7)
    st.pyplot(fig7)

    st.markdown("### Swarm Plot of Sepal Width by Species")
    fig8, ax8 = plt.subplots()
    sns.swarmplot(x='species', y='sepal width (cm)', data=df, ax=ax8)
    st.pyplot(fig8)

elif option == "Model Training":
    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy**: {accuracy:.4f}")
    st.write(f"**Mean Squared Error (MSE)**: {mse:.4f}")
    st.write(f"**Mean Absolute Error (MAE)**: {mae:.4f}")

    st.markdown("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    st.pyplot(fig)

elif option == "Make Predictions":
    st.subheader("🌼 Predict Iris Species")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"The predicted Iris species is: **{prediction[0].capitalize()}**")
