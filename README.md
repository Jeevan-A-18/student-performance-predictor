# 📚 Student Performance Predictor

A beginner-friendly Machine Learning web application that predicts a student's
examination marks based on the number of hours they study — built with Python,
Scikit-learn and Streamlit.

---

## 🗂️ Project Structure

```
student-performance-predictor/
│
├── app.py           ← Streamlit web application (UI)
├── model.py         ← Machine Learning logic
├── dataset.csv      ← Student dataset (study hours + marks)
├── requirements.txt ← Python dependencies
└── README.md        ← This file
```

---

## ⚙️ Installation & Setup

### Step 1 — Make sure Python is installed

Open a terminal and run:

```bash
python --version
```

You need **Python 3.9 or higher**. Download it from [python.org](https://www.python.org/downloads/) if needed.

---

### Step 2 — (Optional but recommended) Create a virtual environment

```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS / Linux:
source venv/bin/activate
```

---

### Step 3 — Install the required libraries

```bash
pip install -r requirements.txt
```

---

### Step 4 — Run the application

```bash
streamlit run app.py
```

Your browser will open automatically at **http://localhost:8501**.

---

## 🧠 Machine Learning Concepts Explained (For Beginners)

This section explains the key ideas behind this project in simple language.

---

### 📌 What is Linear Regression?

**Linear Regression** is one of the simplest Machine Learning algorithms.

It finds a **straight line** that best fits a set of data points.

Think of it like this: you plot all students' study hours on the X-axis and
their marks on the Y-axis. You then draw the best possible straight line
through those dots. That line lets you predict marks for any number of hours.

The equation of the line is:

```
marks = (slope × study_hours) + intercept
```

- **slope** — how many extra marks you gain for each additional hour of study
- **intercept** — the baseline marks (when study hours = 0)

Scikit-learn finds these two numbers automatically during training.

---

### 📌 What are Features and Labels?

| Term        | Meaning                          | In This Project  |
|-------------|----------------------------------|------------------|
| **Feature** | The input given to the model     | `study_hours`    |
| **Label**   | The output the model predicts    | `marks`          |

- **Features (X)** are the things we *know* (study hours).
- **Labels (y)** are the things we *want to predict* (examination marks).

The model learns to map features → labels.

---

### 📌 Why do we split data into Training and Testing sets?

We have a dataset of 40 students. We divide it into two parts:

| Set              | Purpose                                            | Size |
|------------------|----------------------------------------------------|------|
| **Training set** | Used to teach the model (it sees the answers)      | 80 % |
| **Testing set**  | Used to check how well the model learned           | 20 % |

If we tested the model on the same data it trained on, it would be like giving
a student the exact same questions they already practised — not a fair test!

The testing set contains data the model has **never seen before**, giving us
an honest measure of real-world performance.

---

### 📌 What does "Model Training" mean?

Training is the process where the algorithm **looks at the training data** and
**adjusts its internal numbers** (slope and intercept) until the line fits the
data as closely as possible.

In code, this is simply:

```python
model = LinearRegression()
model.fit(X_train, y_train)   # ← training happens here
```

After `.fit()` is done, the model has "learned" from the data and is ready
to make predictions.

---

### 📌 What are MAE and R² Score?

These are **evaluation metrics** — numbers that tell us how good our model is.

#### Mean Absolute Error (MAE)

MAE is the **average difference** between the predicted marks and the actual marks.

```
MAE = average of |predicted_marks − actual_marks|
```

- A **lower MAE is better**.
- MAE = 3 means the model's predictions are off by about 3 marks on average.

#### R² Score (R-Squared)

R² measures **how well the model's line fits the data**.

- **R² = 1.0** → Perfect fit — the model predicts everything exactly right.
- **R² = 0.0** → The model is no better than simply guessing the average marks every time.
- Values **closer to 1.0** are better.

For our simple model using only one feature (study hours), an R² above 0.95
is considered excellent.

---

## 📊 Dataset

The dataset (`dataset.csv`) contains 40 student records with two columns:

| Column        | Description                                |
|---------------|--------------------------------------------|
| `study_hours` | Number of hours the student studied (1–12) |
| `marks`       | Examination marks scored (out of 100)      |

The data includes realistic variation — marks are not perfectly proportional
to hours, just as in real life.

---

## 🚀 Application Sections

| Section              | What you'll find                                               |
|----------------------|----------------------------------------------------------------|
| **Home**             | Project description and overview                               |
| **Predict**          | Slider to enter study hours → see predicted marks instantly    |
| **Visualization**    | Scatter plot showing all data points + the regression line     |
| **Model Evaluation** | MAE, R² Score, training size and testing size                  |

---

## 🛠️ Technologies Used

| Library       | Purpose                        |
|---------------|--------------------------------|
| Python        | Programming language           |
| Pandas        | Loading and handling data      |
| NumPy         | Numerical operations           |
| Scikit-learn  | Linear Regression model        |
| Matplotlib    | Plotting charts                |
| Streamlit     | Building the web interface     |

---

## 💡 Learning Tips

1. Open `model.py` first — all the ML steps are explained in comments.
2. Change the `test_size` in `split_data()` and see how it affects the metrics.
3. Add a few rows to `dataset.csv` and re-run the app to see the model update.
4. Try changing the slider and watch the green star move on the scatter plot!

---

*Happy Learning! 🎓*
