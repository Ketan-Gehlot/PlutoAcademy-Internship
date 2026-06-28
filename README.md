# Heart Disease ML Pipeline

This is an end-to-end Machine Learning project that guides users step-by-step through a complete data science pipeline: from data loading and feature engineering to model training, evaluation, and detailed analysis. It features a clean, professional React frontend and a robust FastAPI backend.

## 🚀 How to Run the Project Locally

### 1. Backend Setup
Navigate to the `backend` directory and set up the Python environment:
```bash
cd backend
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```
The backend API will run on `http://localhost:8000`.

### 2. Frontend Setup
Open a new terminal window, navigate to the `frontend` directory, and start the React app:
```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
The frontend UI will run on `http://localhost:5173`. Open this in your browser to interact with the project!

---

## 🧠 Detailed Project Architecture

Here is a complete, detailed breakdown of exactly what was built and how the project works under the hood.

### 1. The Backend Setup (Python, FastAPI, Scikit-Learn)
A modular ML pipeline (`ml_pipeline.py`) was built to handle the 5 core machine learning steps:

- **Step 1: Load & Preprocess:** The backend loads the `heart.csv` file using Pandas. It extracts the `target` column (what we want to predict), and splits the data into 80% training data and 20% testing data. Crucially, `StandardScaler` is applied to normalize the numerical features, which is mandatory for distance-based algorithms like KNN to work correctly.
- **Step 2: Feature Engineering:** The code calculates the Pearson Correlation between every feature and the target. It also trains a quick Random Forest model to extract "Feature Importances." Using this data, the code automatically identifies the 2 *least* important features and drops them to reduce noise and simplify the model.
- **Step 3: Training:** The backend initializes and trains three separate machine learning models simultaneously on the scaled training data:
  1. `Logistic Regression` (Great for linear relationships and interpretability)
  2. `Random Forest Classifier` (An ensemble of decision trees, very robust)
  3. `K-Nearest Neighbors (KNN)` (Predicts based on the closest data points)
- **Step 4: Evaluation:** All three models are tested on the 20% unseen test data. The code calculates the Accuracy, Precision, Recall, and F1-Score to rigorously determine which model performed the best.
- **Step 5: Analysis:** The pipeline automatically identifies the "Best Model" based on the highest F1-Score and generates a Confusion Matrix to show true positives, true negatives, false positives, and false negatives.

**API Layer:** `main.py` uses **FastAPI** to expose these 5 steps as REST API endpoints (`/api/step1_load`, `/api/step2_features`, etc.) so the frontend can securely communicate with the machine learning code.

### 2. The Frontend Setup (React, Vite, Vanilla CSS)
A React single-page application built with Vite acts as the user interface, strictly adhering to a **clean, professional, light-mode** design:

- **Design System:** All styling is handled via custom pure CSS (`index.css`) rather than relying on bloated frameworks. It utilizes a minimalist layout with a sticky sidebar acting as a "Stepper" to visually track progress.
- **Color Palette:** Built with clean whites, soft slate grays (`#f8fafc`), and professional blues (`#2563eb`) to ensure a premium corporate aesthetic.
- **State Management:** `App.jsx` handles fetching data sequentially from the FastAPI backend. Each step unlocks only when the previous ML task completes successfully.
- **Data Visualization:** The application integrates the `recharts` library to dynamically generate beautiful, interactive bar charts for the Feature Importance ranking and the Model Metrics comparison.
- **Confusion Matrix UI:** A custom CSS grid visually renders the final Confusion Matrix with color-coded success (green) and error (red) boxes, making it incredibly easy for a non-technical person to understand the model's true performance.
