# 🌸 Iris Species Classification App

This is a **Streamlit web application** for classifying Iris flower species using a **Random Forest classifier**. It provides:

- Dataset exploration
- Multiple data visualizations
- Model training with evaluation metrics
- Real-time predictions using user input sliders

---

## 📦 Features

- ✅ View the Iris dataset
- 📊 Visualize with pairplots, boxplots, heatmaps, and more
- 🤖 Train and evaluate a Random Forest Classifier
- 🧪 Metrics: Accuracy, MSE, MAE, Confusion Matrix, Classification Report
- 🌼 Predict the Iris species using interactive sliders

---

## 📁 File Structure

```

iris\_species\_classifier/
├── iris\_app.py       # Streamlit application
├── README.md         # Project documentation
├── requirements.txt  # Dependencies (optional)

````

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/iris-species-classifier.git
cd iris-species-classifier
````

### 2. Install dependencies

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install streamlit scikit-learn pandas matplotlib seaborn
```

### 3. Run the app

```bash
streamlit run iris_app.py
```

---

## 📷 Preview
![Sc![Screenshot 2025-06-13 212551](https://github.com/user-attachments/assets/cdf114e2-dff0-4c39-9855-00a450bbdd18)
reenshot 2025-06-13 212610](https://github.com/user-attachments/assets/34ca0e0d-4812-424f-ab30-91f0bb20d130)
![Screenshot 2025-06-13 212449](https://github.com/user-attachments/assets/8f49bd03-6f94-479f-8bb3-9957c72d77b3)
![Screenshot 2025-06-13 212449](https://github.com/user-attachments/assets/b70d062d-3769-465e-ac72-3dd430772d08)
![Screenshot 2025-06-13 212423](https://github.com/user-attachments/assets/8bfe0eee-729c-4e7b-831f-3e2d0e9f262b)


---

## 🧠 Dataset

This app uses the [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), which contains 150 samples from three species of Iris (setosa, versicolor, virginica), with 4 features each:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

---

## 📈 Model

The app uses a **Random Forest Classifier** from `scikit-learn`, trained on standardized features. Evaluation includes:

* Accuracy Score
* Mean Squared Error
* Mean Absolute Error
* Confusion Matrix
* Classification Report

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a PR.

---

## 📄 License

MIT License. Feel free to use, modify, and share.

---

## 👤 Author

Made with ❤️ by Jayasimma D

```

