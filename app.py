import streamlit as st
import pandas as pd
import numpy as np
import os
from automl_pipeline import run_automl_pipeline  # modularized function
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide", 
    page_title="AutoPilotML Pipeline", 
    page_icon="🚀", 
    initial_sidebar_state="expanded"
)

# --- Enhanced Page Styling ---
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 300% 300%;
        animation: gradient 8s ease infinite;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Header layout responsive design */
    .header-container {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 2rem;
    }
    
    .control-panel-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        min-width: 250px;
        height: fit-content;
    }
    
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;
        }
        .control-panel-header {
            min-width: 100%;
        }
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Minimized sidebar styling */
    .css-1cypcdb {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Sidebar collapse button area styling - Multiple selectors */
    .css-1rs6os, .css-1544g2n, [data-testid="collapsedControl"], 
    .css-17eq0hr, .css-164nlkn, .css-1v8rj3v, .stSidebar > div > div {
        position: relative;
    }
    
    /* Add Control Panel text next to collapse arrow - Multiple approaches */
    .css-1rs6os::after, .css-1544g2n::after, [data-testid="collapsedControl"]::after,
    .css-17eq0hr::after, .css-164nlkn::after, .css-1v8rj3v::after {
        content: "Control Panel";
        color: white;
        font-size: 11px;
        font-weight: 600;
        position: absolute;
        left: 25px;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(102, 126, 234, 0.9);
        padding: 4px 8px;
        border-radius: 12px;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        z-index: 999;
    }
    
    /* Streamlit sidebar toggle button styling */
    button[kind="secondary"] {
        position: relative !important;
    }
    
    button[kind="secondary"]::after {
        content: "⚙️ Control Panel";
        color: white;
        font-size: 10px;
        font-weight: 600;
        position: absolute;
        left: calc(100% + 10px);
        top: 50%;
        transform: translateY(-50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4px 8px;
        border-radius: 8px;
        white-space: nowrap;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Custom metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 0.5rem 0;
        position: relative;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 320px;
        background: rgba(44, 62, 80, 0.98);
        color: white;
        text-align: left;
        border-radius: 12px;
        padding: 16px;
        position: absolute;
        z-index: 9999;
        top: -10px;
        left: 50%;
        margin-left: -160px;
        opacity: 0;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        border: 2px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        bottom: -10px;
        left: 50%;
        margin-left: -10px;
        border-width: 10px;
        border-style: solid;
        border-color: rgba(44, 62, 80, 0.98) transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
        top: -20px;
    }
    
    /* Responsive tooltip positioning */
    @media (max-width: 768px) {
        .tooltip .tooltiptext {
            width: 280px;
            margin-left: -140px;
            font-size: 0.9rem;
        }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        transform: translateY(0);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        justify-content: flex-start;
        overflow-x: auto;
        white-space: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 8px 16px;
        margin: 0 1px;
        min-width: 80px;
        text-align: center;
        font-size: 0.9rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-1px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
    }
    
    /* Loading spinner custom */
    .stSpinner {
        color: #4ecdc4;
    }
    
    /* Custom warning styling */
    .stWarning {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        border-radius: 10px;
        padding: 1rem;
        color: #2c3e50;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Enhanced App Header ---
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57); background-size: 300% 300%; animation: gradient 8s ease infinite; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 1rem;">
        <h1 style="color: white; font-size: 2.5rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🚀 AutoPilotML Pipeline</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem; font-weight: 300;">Advanced Machine Learning Automation Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Enhanced Sidebar ---
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">⚙️ Control Panel</h2>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Configure your ML pipeline</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    help="Upload your dataset in CSV format. Supported formats: .csv"
)

# --- Enhanced Model Configuration ---
st.sidebar.header("🔧 Model Configuration")

with st.sidebar.expander("🎯 Model Selection Strategy", expanded=True):
    model_choice = st.selectbox(
        "Strategy", 
        ["gridsearch", "accuracy", "utility"], 
        help="""Choose your model selection strategy:

🔬 **gridsearch**: Full hyperparameter tuning (recommended)
- Most thorough optimization
- Best overall performance
- Longest training time

🎯 **accuracy**: Pure performance focus
- Selects highest scoring model
- No time considerations
- Good for maximum accuracy

⚡ **utility**: Balanced approach
- 70% performance + 30% speed
- Practical for production use
- May choose faster models over marginal accuracy gains"""
    )

with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
    force_clean_regression = st.checkbox(
        "Force Clean Regression Targets", 
        help="Force clean regression targets even with high bad ratio"
    )
    
    selected_class_for_roc = st.text_input(
        "Class for ROC Curve (multiclass only)", 
        help="Enter class name for multiclass ROC curves"
    )

# --- Info Panel ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); padding: 1rem; border-radius: 10px; text-align: center;">
        <h4 style="color: white; margin: 0;">📊 Pipeline Features</h4>
        <ul style="color: rgba(255,255,255,0.9); text-align: left; margin: 0.5rem 0 0 0;">
            <li>🤖 12+ ML Algorithms</li>
            <li>🔍 SHAP Explainability</li>
            <li>📈 Auto Visualization</li>
            <li>⭐ Performance Rating</li>
            <li>💾 Model Export</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Main Content Area ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Dataset info card
    st.markdown(
        f"""
        <div class="metric-card tooltip">
            <h3 style="color: white; margin: 0 0 1rem 0;">📄 Dataset Overview</h3>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div class="tooltip" style="cursor: pointer;">
                    <h4 style="color: #4ecdc4; margin: 0;">{df.shape[0]:,}</h4>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Rows</p>
                    <span class="tooltiptext">
                        <strong>Dataset Rows:</strong> Total number of data samples/records in your dataset. 
                        More rows generally lead to better model performance, especially for complex patterns.
                    </span>
                </div>
                <div class="tooltip" style="cursor: pointer;">
                    <h4 style="color: #ff6b6b; margin: 0;">{df.shape[1]:,}</h4>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Columns</p>
                    <span class="tooltiptext">
                        <strong>Dataset Columns:</strong> Total number of features/variables in your dataset. 
                        Includes both input features and the target variable you want to predict.
                    </span>
                </div>
                <div class="tooltip" style="cursor: pointer;">
                    <h4 style="color: #feca57; margin: 0;">{df.isnull().sum().sum():,}</h4>
                    <p style="color: rgba(255,255,255,0.8); margin: 0;">Missing Values</p>
                    <span class="tooltiptext">
                        <strong>Missing Values:</strong> Total number of empty/null cells in your dataset. 
                        AutoPilotML automatically handles missing values using intelligent imputation strategies.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # --- Enhanced Target Column Selection ---
    st.markdown("### 🎯 Target Configuration")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        target_column = st.selectbox(
            "Select Target Column", 
            df.columns,
            help="Choose the column you want to predict"
        )
    
    with col2:
        if target_column:
            unique_values = df[target_column].nunique()
            st.metric("Unique Values", unique_values)

    # Show target distribution
    if target_column:
        st.markdown("#### 📊 Target Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            value_counts = df[target_column].value_counts().head(10)
            st.bar_chart(value_counts)
        
        with col2:
            st.write("**Top 10 Values:**")
            for idx, (value, count) in enumerate(value_counts.items()):
                percentage = (count / len(df)) * 100
                st.write(f"• **{value}**: {count:,} ({percentage:.1f}%)")

    # --- Enhanced Run Button ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_pipeline = st.button(
            "🚀 Launch AutoPilotML Pipeline", 
            use_container_width=True,
            type="primary"
        )
    
    if run_pipeline:
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🤖 AutoPilotML is training your models... Please wait"):
            try:
                # Update progress
                progress_bar.progress(10)
                status_text.text("🔍 Analyzing dataset...")
                
                # Prepare optional parameters
                roc_class = selected_class_for_roc if selected_class_for_roc.strip() else None
                
                progress_bar.progress(20)
                status_text.text("⚙️ Configuring pipeline...")
                
                # Run the AutoML pipeline
                results = run_automl_pipeline(
                    df=df, 
                    target_col=target_column,
                    model_choice=model_choice,
                    force_clean_regression=force_clean_regression,
                    selected_class_for_roc=roc_class
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Pipeline complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # --- Enhanced Results Display ---
                st.balloons()  # Celebration effect!
                
                st.markdown(
                    """
                    <div style="background: linear-gradient(90deg, #56ab2f, #a8e6cf); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                        <h2 style="color: white; margin: 0;">🎉 Pipeline Execution Complete!</h2>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Your AutoPilotML model is ready!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Enhanced metrics display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card tooltip" style="text-align: center;">
                            <h3 style="color: #4ecdc4; margin: 0;">🎯 Task Type</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{results['task'].capitalize()}</h2>
                            <span class="tooltiptext">
                                <strong>Task Type:</strong> Indicates whether this is a classification problem 
                                (predicting categories) or regression problem (predicting continuous values). 
                                AutoPilotML automatically detects the appropriate task type based on your target variable.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card tooltip" style="text-align: center;">
                            <h3 style="color: #ff6b6b; margin: 0;">🏆 Best Model</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{results['best_model']}</h2>
                            <span class="tooltiptext">
                                <strong>Best Model:</strong> The machine learning algorithm that performed best 
                                on your dataset. AutoPilotML tested 12+ different algorithms and selected this one 
                                based on cross-validation performance and your chosen strategy.
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col3:
                    rating = results['overall_rating']
                    rating_color = "#ff6b6b" if rating < 5 else "#feca57" if rating < 8 else "#4ecdc4"
                    rating_description = (
                        "Poor performance - may need more data or feature engineering" if rating < 5 
                        else "Good performance - model is reliable for most use cases" if rating < 8 
                        else "Excellent performance - model is highly accurate and ready for production"
                    )
                    st.markdown(
                        f"""
                        <div class="metric-card tooltip" style="text-align: center;">
                            <h3 style="color: {rating_color}; margin: 0;">⭐ Overall Rating</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{rating:.1f}/10</h2>
                            <span class="tooltiptext">
                                <strong>Overall Rating:</strong> A comprehensive score (1-10) based on model accuracy, 
                                cross-validation stability, and data quality. <br><br>
                                <strong>Your Score:</strong> {rating_description}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- Performance Metrics ---
                st.subheader("📊 Model Performance")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.write("**Performance Metrics:**")
                    for key, value in results['metrics'].items():
                        if isinstance(value, float):
                            st.write(f"• **{key.upper()}**: {value:.4f}")
                        else:
                            st.write(f"• **{key.upper()}**: {value}")

                with metrics_col2:
                    st.write("**Model Summary:**")
                    st.write(results['summary'])

                # --- Display Logs ---
                with st.expander("📋 Detailed Training Logs", expanded=False):
                    for log in results['logs']:
                        st.text(log)

                # --- Display Plots ---
                if results['plots']:
                    st.subheader("📈 Visualizations")
                    
                    # Create consistent tab labels with proper spacing
                    num_plots = len(results['plots'])
                    tab_labels = []
                    for i in range(num_plots):
                        tab_labels.append(f"Plot {i+1:02d}")  # Zero-padded for alignment
                    
                    # Create tabs with consistent spacing
                    plot_tabs = st.tabs(tab_labels)
                    
                    for i, (tab, plot) in enumerate(zip(plot_tabs, results['plots'])):
                        with tab:
                            # Create a container for better plot sizing
                            plot_container = st.container()
                            with plot_container:
                                # Set figure size before displaying - smaller plots
                                if hasattr(plot, 'set_size_inches'):
                                    plot.set_size_inches(6, 4)  # Reduced from (8, 5)
                                
                                # Display plot with custom width
                                st.pyplot(plot, use_container_width=False)

                # --- SHAP Analysis ---
                if results['shap_analysis'] and results['shap_analysis'].get('feature_importance') is not None:
                    st.subheader("🔍 SHAP Feature Importance Analysis")
                    
                    shap_col1, shap_col2 = st.columns(2)
                    
                    with shap_col1:
                        st.write("**Top 10 Most Important Features:**")
                        feature_importance = results['shap_analysis']['feature_importance']
                        feature_names = results['shap_analysis'].get('feature_names', [f"feature_{i}" for i in range(len(feature_importance))])
                        
                        # Create a simple bar chart for feature importance
                        top_indices = np.argsort(feature_importance)[-10:][::-1]
                        top_features = [feature_names[i] for i in top_indices]
                        top_importance = [feature_importance[i] for i in top_indices]
                        
                        importance_df = pd.DataFrame({
                            'Feature': top_features,
                            'Importance': top_importance
                        })
                        st.bar_chart(importance_df.set_index('Feature'))
                    
                    with shap_col2:
                        st.write("**SHAP Analysis Summary:**")
                        for log in results['shap_analysis'].get('analysis_logs', []):
                            st.text(log)

                # --- Model Download ---
                st.subheader("💾 Download Trained Model")
                
                if os.path.exists(results['model_path']):
                    with open(results['model_path'], "rb") as f:
                        st.download_button(
                            label="⬇️ Download Best Model (.pkl)",
                            data=f.read(),
                            file_name=f"{results['best_model'].replace(' ', '_')}_model.pkl",
                            mime="application/octet-stream"
                        )
                    st.success(f"✅ Model saved at: {results['model_path']}")
                else:
                    st.error("❌ Model file not found")

                # --- Model Scores Comparison ---
                if results['model_scores']:
                    st.subheader("🏅 All Model Scores Comparison")
                    scores_df = pd.DataFrame(list(results['model_scores'].items()), 
                                           columns=['Model', 'Score'])
                    scores_df = scores_df.sort_values('Score', ascending=False)
                    st.bar_chart(scores_df.set_index('Model'))

            except Exception as e:
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                st.error("Please check your dataset and try again.")
                st.exception(e)  # Show full traceback for debugging

else:
    # --- Welcome Section ---
    st.markdown(
        """
        <div class="metric-card" style="text-align: center; margin: 2rem 0;">
            <h2 style="color: white; margin: 0 0 1rem 0;">👋 Welcome to AutoPilotML!</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; line-height: 1.6;">
                Your AI-powered machine learning automation platform. Simply upload your dataset 
                and let AutoPilotML handle the rest!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Features showcase
    st.markdown("### 🚀 What AutoPilotML Can Do")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #4ecdc4; margin: 0 0 1rem 0;">🤖 Smart Automation</h3>
                <ul style="color: rgba(255,255,255,0.8); text-align: left; padding-left: 1rem;">
                    <li>Auto feature detection</li>
                    <li>Intelligent preprocessing</li>
                    <li>Missing value handling</li>
                    <li>Data type optimization</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #ff6b6b; margin: 0 0 1rem 0;">🏆 Model Training</h3>
                <ul style="color: rgba(255,255,255,0.8); text-align: left; padding-left: 1rem;">
                    <li>12+ ML algorithms</li>
                    <li>Hyperparameter tuning</li>
                    <li>Cross-validation</li>
                    <li>Performance optimization</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #feca57; margin: 0 0 1rem 0;">📊 Analysis & Export</h3>
                <ul style="color: rgba(255,255,255,0.8); text-align: left; padding-left: 1rem;">
                    <li>SHAP explainability</li>
                    <li>Interactive visualizations</li>
                    <li>Model comparison</li>
                    <li>Easy model export</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Getting started
    st.markdown("### 🎯 Getting Started")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <ol style="color: white; font-size: 1.1rem; line-height: 1.8;">
                <li><strong>📁 Upload</strong> your CSV dataset using the sidebar</li>
                <li><strong>🎯 Select</strong> your target column (what you want to predict)</li>
                <li><strong>⚙️ Configure</strong> advanced settings if needed</li>
                <li><strong>🚀 Launch</strong> the AutoPilotML pipeline</li>
                <li><strong>📊 Analyze</strong> results and download your trained model</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sample datasets info
    st.markdown("### 📂 Sample Datasets Available")
    sample_datasets = [
        {"name": "🌸 Iris", "description": "Classic flower classification", "features": "4", "rows": "150"},
        {"name": "❤️ Heart Disease", "description": "Medical diagnosis prediction", "features": "13", "rows": "918"},
        {"name": "🏠 Housing", "description": "Price prediction", "features": "13", "rows": "506"},
        {"name": "🏦 Bank Marketing", "description": "Campaign success prediction", "features": "20", "rows": "4521"},
        {"name": "🚗 Car Insurance", "description": "Insurance claim prediction", "features": "7", "rows": "1000"},
        {"name": "📊 Customer Analytics", "description": "Customer behavior analysis", "features": "12", "rows": "2500"},
    ]
    
    for i in range(0, len(sample_datasets), 2):
        col1, col2 = st.columns(2)
        
        if i < len(sample_datasets):
            with col1:
                dataset = sample_datasets[i]
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; border-left: 4px solid #4ecdc4; margin-bottom: 1rem;">
                        <h4 style="color: white; margin: 0;">{dataset['name']}</h4>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">{dataset['description']}</p>
                        <small style="color: rgba(255,255,255,0.5);">{dataset['features']} features • {dataset['rows']} rows</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        if i + 1 < len(sample_datasets):
            with col2:
                dataset = sample_datasets[i + 1]
                st.markdown(
                    f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; border-left: 4px solid #ff6b6b; margin-bottom: 1rem;">
                        <h4 style="color: white; margin: 0;">{dataset['name']}</h4>
                        <p style="color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0;">{dataset['description']}</p>
                        <small style="color: rgba(255,255,255,0.5);">{dataset['features']} features • {dataset['rows']} rows</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    st.info("💡 **Pro Tip**: Start with one of the sample datasets in your `Datasets/` folder to test the pipeline!")
