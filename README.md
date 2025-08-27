🚀 AutoPilotML: End-to-End Automated Machine Learning Pipeline

AutoPilotML is an end-to-end AutoML system built with Python & Streamlit that allows users to upload datasets, automatically preprocess data, train multiple ML models, evaluate performance, and download results — all without writing code.

It bridges the gap between data preprocessing → model training → evaluation → explainability → export, making ML accessible to both beginners and professionals.

✨ Features

📊 Data Input: Upload CSV or use sample datasets with automatic data type detection

🧹 Preprocessing: Missing value handling, encoding, scaling, feature engineering

⚡ Dual Workflow Modes:

Phase 1 → Train all 13 ML algorithms with cross-validation (fast, no hyperparams)

Phase 2 → Select top 3 models and fine-tune with Randomized/GridSearchCV

🏆 Model Selection: Automated comparison with multiple metrics (classification & regression)

📈 Explainability: SHAP values, feature importance, visualizations

💾 Download Options: Export trained model, metrics, and cleaned dataset

👤 User Accounts: Login system with Guest Mode (try without signing up)

🖥️ UI Modes: Normal Mode (simple) & Developer Mode (detailed “Hacker Mode” insights)

🛠️ Tech Stack

Frontend/UI → Streamlit

Backend/ML → Scikit-learn, SHAP, Pandas, NumPy

Database (for logged users) → SQLite / PostgreSQL (optional)

Deployment → Streamlit Cloud / Heroku / AWS

🔄 Workflow

Data Input → Upload dataset

Cleaning & Preprocessing → Handle missing values, encoding, scaling

Phase 1 Training → Train all models (quick cross-validation, no hyperparams)

Phase 2 Training → Select top models, run Randomized/GridSearchCV

Evaluation → Metrics (accuracy, F1, RMSE, R², etc.)

Explainability → SHAP & feature insights

Export → Download trained model + cleaned dataset

📊 Supported Models

✅ Classification (Logistic Regression, RandomForest, SVM, XGBoost, etc.)
✅ Regression (LinearRegression, RandomForestRegressor, GradientBoosting, etc.)
✅ Ensemble Learning (Voting, Stacking for both tasks)

📌 Roadmap (WIP – ~90% Complete)

 Core preprocessing & training pipeline

 Multi-model training & evaluation

 SHAP explainability & visualizations

 Download trained model

 Login system + Guest mode

 Database storage of past runs & scores

 Extended hyperparameter tuning options

 Cloud deployment + API access
