# =============================
#   AutoPilotML Streamlit App
# =============================
# Comments are deliberately added for easy debugging and understanding of the code.
# This file builds the interactive dashboard for AutoPilotML using Streamlit.
# It covers: UI layout, dataset upload, model configuration, pipeline execution,
# results visualization, and model download. Each section is explained below.

# --- Import Required Libraries ---
import streamlit as st  # Main Streamlit library for building web UI
import pandas as pd     # For DataFrame operations and data manipulation
import numpy as np      # For numerical operations and array handling
import os               # For file system operations
from automl_pipeline import run_automl_pipeline  # Our main ML pipeline function
import pickle           # For model serialization and deserialization
import matplotlib.pyplot as plt  # For custom plotting if needed

# --- Streamlit Page Configuration ---
# This sets up the basic properties of the web page
st.set_page_config(
    layout="wide",  # Uses full width of browser
    page_title="AutoPilotML Pipeline",  # Browser tab title
    page_icon="🚀",  # Browser tab icon
    initial_sidebar_state="expanded"  # Sidebar is open by default
)

# --- Enhanced Page Styling ---
# This section injects custom CSS to style the entire application.
# It controls colors, fonts, animations, card styles, tooltips, buttons, etc.
# You can modify colors, sizes, and effects here to customize the look.
st.markdown(
    """
    <style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Main app background and font styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Purple gradient background */
        font-family: 'Inter', sans-serif;  /* Modern font for better readability */
    }
    
    /* Header styling with animated gradient background */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);  /* Multi-color gradient */
        background-size: 300% 300%;  /* Makes gradient larger for animation */
        animation: gradient 8s ease infinite;  /* Animates the gradient colors */
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);  /* Drop shadow for depth */
    }
    
    /* Responsive header layout for different screen sizes */
    .header-container {
        display: flex;  /* Flexbox layout for side-by-side elements */
        gap: 1rem;     /* Space between elements */
        align-items: flex-start;  /* Align items to top */
        margin-bottom: 2rem;
    }
    
    /* Control panel header styling */
    .control-panel-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        min-width: 250px;  /* Minimum width to prevent shrinking */
        height: fit-content;  /* Height adjusts to content */
    }
    
    /* Mobile responsiveness - stacks elements vertically on small screens */
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column;  /* Stack vertically on mobile */
        }
        .control-panel-header {
            min-width: 100%;  /* Full width on mobile */
        }
    }
    
    /* Animation keyframes for the gradient background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }    /* Start position */
        50% { background-position: 100% 50%; } /* Middle position */
        100% { background-position: 0% 50%; }  /* End position (back to start) */
    }
    
    /* Header text styling */
    .main-header h1 {
        color: white;
        font-size: 3rem;     /* Large title text */
        font-weight: 700;    /* Bold weight */
        margin: 0;           /* Remove default margins */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);  /* Text shadow for readability */
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);  /* Slightly transparent white */
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;    /* Light weight for subtitle */
    }
    
    /* Sidebar background styling - targets Streamlit's sidebar classes */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);  /* Dark gradient */
    }
    
    /* Minimized sidebar styling */
    .css-1cypcdb {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Sidebar collapse button styling - multiple selectors for different Streamlit versions */
    .css-1rs6os, .css-1544g2n, [data-testid="collapsedControl"], 
    .css-17eq0hr, .css-164nlkn, .css-1v8rj3v, .stSidebar > div > div {
        position: relative;
    }
    
    /* Custom metric cards styling - these are the main info boxes */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Purple gradient background */
        padding: 1.5rem;          /* Inner spacing */
        border-radius: 12px;      /* Rounded corners */
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);  /* Drop shadow for depth */
        border: 1px solid rgba(255,255,255,0.1); /* Subtle border */
        backdrop-filter: blur(10px);  /* Blur effect for modern look */
        margin: 0.5rem 0;         /* Vertical spacing between cards */
        position: relative;       /* For positioning child elements */
        transition: all 0.3s ease;  /* Smooth animations on hover */
        cursor: pointer;          /* Shows it's interactive */
    }
    
    /* Hover effect for metric cards - makes them "lift up" */
    .metric-card:hover {
        transform: translateY(-5px);  /* Moves card up slightly */
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);  /* Larger shadow */
        border: 1px solid rgba(255,255,255,0.3);  /* Brighter border */
    }
    
    /* Tooltip system - shows helpful information on hover */
    .tooltip {
        position: relative;      /* Allows positioning of tooltip */
        display: inline-block;   /* Allows hover detection */
    }
    
    /* Tooltip text box styling */
    .tooltip .tooltiptext {
        visibility: hidden;      /* Hidden by default */
        width: 320px;           /* Fixed width for consistency */
        background: rgba(44, 62, 80, 0.98);  /* Dark semi-transparent background */
        color: white;           /* White text */
        text-align: left;       /* Left-aligned text */
        border-radius: 12px;    /* Rounded corners */
        padding: 16px;          /* Inner spacing */
        position: absolute;     /* Positioned relative to parent */
        z-index: 9999;         /* Appears above everything else */
        top: -10px;            /* Position above the element */
        left: 50%;             /* Center horizontally */
        margin-left: -160px;   /* Offset to truly center (half of width) */
        opacity: 0;            /* Transparent by default */
        transition: all 0.3s ease;  /* Smooth fade in/out */
        font-size: 0.95rem;    /* Readable font size */
        line-height: 1.5;      /* Good line spacing */
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);  /* Drop shadow */
        border: 2px solid rgba(255,255,255,0.2);  /* Subtle border */
        backdrop-filter: blur(10px);  /* Blur background behind tooltip */
    }
    
    /* Tooltip arrow pointing downward */
    .tooltip .tooltiptext::after {
        content: "";           /* Empty content for the arrow */
        position: absolute;    /* Positioned relative to tooltip */
        bottom: -10px;        /* Below the tooltip */
        left: 50%;            /* Centered horizontally */
        margin-left: -10px;   /* Offset to center the arrow */
        border-width: 10px;   /* Size of the arrow */
        border-style: solid;  /* Solid border */
        border-color: rgba(44, 62, 80, 0.98) transparent transparent transparent;  /* Arrow color */
    }
    
    /* Show tooltip on hover */
    .tooltip:hover .tooltiptext {
        visibility: visible;   /* Make visible */
        opacity: 1;           /* Fully opaque */
        top: -20px;          /* Move up slightly for better positioning */
    }
    
    /* Mobile responsive tooltip adjustments */
    @media (max-width: 768px) {
        .tooltip .tooltiptext {
            width: 280px;        /* Smaller width for mobile */
            margin-left: -140px; /* Adjust centering for new width */
            font-size: 0.9rem;   /* Slightly smaller text */
        }
    }
    
    /* Button styling for all Streamlit buttons */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);  /* Gradient background */
        color: white;            /* White text */
        border: none;           /* Remove default border */
        padding: 0.75rem 2rem;  /* Inner spacing */
        border-radius: 25px;    /* Rounded button */
        font-weight: 600;       /* Semi-bold text */
        font-size: 1.1rem;      /* Larger text */
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);  /* Drop shadow */
        transition: all 0.3s ease;  /* Smooth hover effects */
        transform: translateY(0);    /* Starting position for animation */
    }
    
    /* Button hover effect */
    .stButton > button:hover {
        transform: translateY(-2px);  /* Lift up on hover */
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);  /* Larger shadow */
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);  /* Reverse gradient */
    }
    
    /* File uploader widget styling */
    .stFileUploader {
        background: rgba(255,255,255,0.05);  /* Subtle white overlay */
        border-radius: 10px;                 /* Rounded corners */
        padding: 1rem;                       /* Inner spacing */
        border: 2px dashed rgba(255,255,255,0.3);  /* Dashed border for drag-drop area */
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);  /* Green gradient */
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);  /* Red gradient */
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;                /* Space between tabs */
        justify-content: flex-start;  /* Align tabs to left */
        overflow-x: auto;        /* Allow horizontal scrolling if needed */
        white-space: nowrap;     /* Keep tabs on one line */
    }
    
    /* Individual tab styling */
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);  /* Purple gradient */
        border-radius: 8px;      /* Rounded corners */
        color: white;           /* White text */
        font-weight: 600;       /* Semi-bold */
        padding: 8px 16px;      /* Inner spacing */
        margin: 0 1px;          /* Small gap between tabs */
        min-width: 80px;        /* Minimum width for consistency */
        text-align: center;     /* Center text */
        font-size: 0.9rem;      /* Slightly smaller text */
        transition: all 0.3s ease;  /* Smooth hover effects */
    }
    
    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);  /* Reverse gradient on hover */
        transform: translateY(-1px);  /* Slight lift effect */
    }
    
    /* Active/selected tab styling */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);  /* Teal gradient for active */
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);  /* Glowing shadow */
    }
    
    /* Loading spinner color customization */
    .stSpinner {
        color: #4ecdc4;  /* Teal color for loading spinner */
    }
    
    /* Warning message styling */
    .stWarning {
        background: linear-gradient(90deg, #f7971e, #ffd200);  /* Orange to yellow gradient */
        border-radius: 10px;
        padding: 1rem;
        color: #2c3e50;    /* Dark text for readability */
        font-weight: 600;  /* Semi-bold text */
    }
    
    /* Hide the load buttons since cards are clickable */
    button[data-testid="baseButton-secondary"][aria-label*="Load"], 
    button[data-testid="baseButton-secondary"][title*="Load"] {
        display: none !important;
    }
    
    /* Enhanced card hover effects */
    div[onclick] {
        transition: all 0.3s ease;
    }
    
    div[onclick]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Hide load buttons visually but keep them functional */
    button[key^="load_btn_"] {
        display: none !important;
        position: absolute;
        visibility: hidden;
        opacity: 0;
        width: 100% !important;
        pointer-events: none;
    }
    
    /* Make all buttons full width */
    .stButton > button {
        width: 100% !important;
        margin: 0 !important;
    }
    
    /* Make dataset cards smaller and more compact */
    .dataset-card {
        padding: 1rem !important;
        margin-bottom: 0.8rem !important;
        min-height: 120px;
        max-height: 150px;
    }
    
    .dataset-card h3 {
        font-size: 1.1rem !important;
        margin: 0 0 0.3rem 0 !important;
    }
    
    .dataset-card p {
        font-size: 0.85rem !important;
        margin: 0.2rem 0 !important;
        line-height: 1.3 !important;
    }
    
    /* Hover effect for dataset cards */
    .dataset-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        border-left-width: 6px !important;
    }
    
    /* Workflow dropdown in sidebar */
    .workflow-dropdown {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
        width: 100%;
    }
    
    .workflow-dropdown:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.4);
    }
    
    .workflow-toggle {
        padding: 12px 16px;
        cursor: pointer;
        user-select: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
        color: white;
        font-size: 0.9rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .workflow-toggle:hover {
        background: rgba(255,255,255,0.1);
    }
    
    .workflow-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease;
        background: rgba(255,255,255,0.05);
        border-radius: 0 0 10px 10px;
        margin-top: 2px;
    }
    
    .workflow-content.expanded {
        max-height: 400px;
        padding: 16px;
    }
    
    .workflow-step {
        margin-bottom: 12px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .workflow-step:last-child {
        border-bottom: none;
        margin-bottom: 0;
    }
    
    .workflow-step h4 {
        color: #feca57;
        margin: 0 0 4px 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .workflow-step p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 0.75rem;
        line-height: 1.4;
    }
    
    .workflow-arrow {
        transition: transform 0.3s ease;
        color: rgba(255,255,255,0.8);
    }
    
    .workflow-arrow.rotated {
        transform: rotate(180deg);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main App Header ---
# This creates the main title banner at the top of the page with animated gradient background
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57); background-size: 300% 300%; animation: gradient 8s ease infinite; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-bottom: 1rem;">
        <h1 style="color: black; font-size: 2.5rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🚀 AutoPilotML Pipeline</h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0.5rem; font-weight: 300;">Your Advanced Machine Learning Pipeline Without Any Leakage</p>
    </div>
    """,
    unsafe_allow_html=True
)



# --- Sidebar Layout ---
# The sidebar contains all user controls and configuration options

# --- Control Panel Section ---
# This displays an informational header in the sidebar explaining what the controls do
st.sidebar.markdown(
    """
    <div style="background: linear-gradient(135deg, #45b7d1 0%, #96ceb4 100%); padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
        <h3 style="color: black; margin: 0 0 0.5rem 0; font-size: 1.3rem; font-weight: 600;">🛠️ Control Panel</h3>
        <p style="color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0;">Configure your ML pipeline settings and manage your workflow from here.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Workflow Dropdown in Sidebar ---
# This creates a dropdown in the sidebar explaining the workflow
st.sidebar.markdown(
    """
    <div class="workflow-dropdown" id="workflowDropdown">
        <div class="workflow-toggle" onclick="toggleWorkflow()">
            <span>🔄 How it Works</span>
            <span class="workflow-arrow" id="workflowArrow">▼</span>
        </div>
        <div class="workflow-content" id="workflowContent">
            <div class="workflow-step">
                <h4>📊 Data Input</h4>
                <p>Upload CSV or select sample dataset with automatic data type detection</p>
            </div>
            <div class="workflow-step">
                <h4>🔧 Preprocessing</h4>
                <p>Feature encoding, scaling, and intelligent missing value handling</p>
            </div>
            <div class="workflow-step">
                <h4>🤖 Model Training</h4>
                <p>Tests 12+ ML algorithms with hyperparameter tuning via GridSearchCV</p>
            </div>
            <div class="workflow-step">
                <h4>🏆 Model Selection</h4>
                <p>Compares performance using cross-validation and selects best model</p>
            </div>
            <div class="workflow-step">
                <h4>📈 Analysis & Export</h4>
                <p>SHAP explainability, visualizations, and downloadable trained model</p>
            </div>
        </div>
    </div>
    
    <script>
    function toggleWorkflow() {
        const content = document.getElementById('workflowContent');
        const arrow = document.getElementById('workflowArrow');
        
        if (content.classList.contains('expanded')) {
            content.classList.remove('expanded');
            arrow.classList.remove('rotated');
        } else {
            content.classList.add('expanded');
            arrow.classList.add('rotated');
        }
    }
    </script>
    """,
    unsafe_allow_html=True
)

# --- Dataset Upload Section ---
# File uploader widget that accepts CSV files
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type=["csv"],  # Only allows CSV files
    help="Upload your dataset in CSV format. Supported formats: .csv"
)

# --- Model Configuration Section ---
# This section lets users choose how the ML pipeline selects the best model
st.sidebar.header("🔧 Model Configuration")

# Model Selection Strategy dropdown with detailed explanations
with st.sidebar.expander("🎯 Model Selection Strategy", expanded=True):
    model_choice = st.selectbox(
        "Strategy", 
        ["gridsearch", "accuracy", "utility"],  # Three different strategies
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

# Advanced Settings for power users
with st.sidebar.expander("⚙️ Advanced Settings", expanded=True):
    # Option to force clean regression targets even if data quality is questionable
    force_clean_regression = st.checkbox(
        "Force Clean Regression Targets", 
        help="Force clean regression targets even with high bad ratio"
    )
    
    # For multiclass classification, specify which class to use for ROC curves
    selected_class_for_roc = st.text_input(
        "Class for ROC Curve (multiclass only)", 
        help="Enter class name for multiclass ROC curves"
    )

# --- Info Panel ---
# This section showcases the key features of AutoPilotML to inform users
st.sidebar.markdown("---")  # Horizontal line separator
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

# =============================
#   Main Content Area
# =============================
# This section displays different content based on whether a file has been uploaded

# --- Case 1: Dataset Uploaded or Loaded from Samples---
# If user has uploaded a file OR loaded from sample datasets, show the main dashboard and analysis tools
if uploaded_file is not None or (hasattr(st.session_state, 'dataset_loaded_from_sample') and st.session_state.dataset_loaded_from_sample and st.session_state.loaded_df is not None):
    # Read the CSV file into a pandas DataFrame for analysis
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        df = st.session_state.loaded_df
        dataset_name = st.session_state.selected_dataset
    
    # --- Dataset Overview Section ---
    # Display summary statistics about the uploaded dataset
    st.markdown(
        f"""
        <div style="text-align: left; margin-bottom: 1.5rem;">
            <h2 style="color: white; font-size: 2rem; font-weight: 700; margin: 0 0 1rem 0;">📄 {dataset_name} - Dataset Overview</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Three metric cards showing key dataset statistics
    # Uses f-string to insert actual dataset values into the HTML
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
            <div class="metric-card tooltip" style="flex: 1; min-width: 200px; max-width: 300px; text-align: center;">
                <h3 style="color: #4ecdc4; margin: 0;">Rows</h3>
                <h2 style="color: #4ecdc4; margin: 0.5rem 0 0 0;">{df.shape[0]:,}</h2>
                <span class="tooltiptext">
                    <strong>Dataset Rows:</strong> Total number of data samples/records in your dataset.<br>
                    More rows generally lead to better model performance, especially for complex patterns.
                </span>
            </div>
            <div class="metric-card tooltip" style="flex: 1; min-width: 200px; max-width: 300px; text-align: center;">
                <h3 style="color: #ff6b6b; margin: 0;">Columns</h3>
                <h2 style="color: #ff6b6b; margin: 0.5rem 0 0 0;">{df.shape[1]:,}</h2>
                <span class="tooltiptext">
                    <strong>Dataset Columns:</strong> Total number of features/variables in your dataset.<br>
                    Includes both input features and the target variable you want to predict.
                </span>
            </div>
            <div class="metric-card tooltip" style="flex: 1; min-width: 200px; max-width: 300px; text-align: center;">
                <h3 style="color: #feca57; margin: 0;">Missing Values</h3>
                <h2 style="color: #feca57; margin: 0.5rem 0 0 0;">{df.isnull().sum().sum():,}</h2>
                <span class="tooltiptext">
                    <strong>Missing Values:</strong> Total number of empty/null cells in your dataset.<br>
                    AutoPilotML automatically handles missing values using intelligent imputation strategies.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # --- Dataset Preview Table ---
    # Shows the first 10 rows so users can see their data structure
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)  # Full width table

    # --- Target Column Selection ---
    # Users must specify which column they want to predict (the target variable)
    st.markdown("### 🎯 Target Configuration")
    col1, col2 = st.columns([3, 1])  # 3:1 ratio layout
    
    with col1:
        # Dropdown with all column names from the dataset
        # If loaded from sample, pre-select the target hint
        target_index = 0
        if hasattr(st.session_state, 'target_hint') and st.session_state.target_hint and st.session_state.target_hint in df.columns:
            target_index = list(df.columns).index(st.session_state.target_hint)
            
        target_column = st.selectbox(
            "Select Target Column", 
            df.columns,  # All columns as options
            index=target_index,
            help="Choose the column you want to predict"
        )
    
    with col2:
        # Show number of unique values in selected target column
        if target_column:
            unique_values = df[target_column].nunique()
            st.metric("Unique Values", unique_values)

    # --- Target Distribution Analysis ---
    # Shows distribution of values in the target column to help users understand their data
    if target_column:
        st.markdown("#### 📊 Target Distribution")
        col1, col2 = st.columns(2)  # Side-by-side layout
        
        with col1:
            # Bar chart showing frequency of each value
            value_counts = df[target_column].value_counts().head(10)  # Top 10 most common values
            st.bar_chart(value_counts)
        
        with col2:
            # Text breakdown showing percentages
            st.write("**Top 10 Values:**")
            for idx, (value, count) in enumerate(value_counts.items()):
                percentage = (count / len(df)) * 100  # Calculate percentage
                st.write(f"• **{value}**: {count:,} ({percentage:.1f}%)")
        st.write("If one value dominates, it may signal class imbalance — AutoPilotML handles this automatically during training.")

    # --- Launch Pipeline Button ---
    # Main action button to start the machine learning process
    st.markdown("---")  # Horizontal separator
    col1, col2, col3 = st.columns([1, 2, 1])  # Center the button
    with col2:
        run_pipeline = st.button(
            "🚀 Launch AutoPilotML Pipeline", 
            use_container_width=True,  # Button fills column width
            type="primary"  # Makes button prominent
        )
    
    # =============================
    #   Pipeline Execution & Results
    # =============================
    # This section runs when the user clicks the "Launch Pipeline" button
    if run_pipeline:
        # Create progress indicators to show pipeline status
        progress_bar = st.progress(0)  # Progress bar from 0-100%
        status_text = st.empty()       # Text that updates with current step
        
        # Show spinner animation while processing
        with st.spinner("🤖 AutoPilotML is training your models... Please wait"):
            try:
                # --- Step 1: Initial Setup ---
                progress_bar.progress(10)
                status_text.text("🔍 Analyzing dataset...")
                
                # --- Step 2: Prepare Parameters ---
                # Convert user inputs to format expected by pipeline
                roc_class = selected_class_for_roc if selected_class_for_roc.strip() else None
                
                progress_bar.progress(20)
                status_text.text("⚙️ Configuring pipeline...")
                
                # --- Step 3: Run the ML Pipeline ---
                # This is the main function that does all the machine learning work
                results = run_automl_pipeline(
                    df=df,                                    # Dataset
                    target_col=target_column,                 # What to predict
                    model_choice=model_choice,                # Strategy for model selection
                    force_clean_regression=force_clean_regression,  # Advanced setting
                    selected_class_for_roc=roc_class          # ROC curve class
                )
                
                # --- Step 4: Pipeline Complete ---
                progress_bar.progress(100)
                status_text.text("✅ Pipeline complete!")
                
                # Remove progress indicators
                progress_bar.empty()
                status_text.empty()

                # --- Results Celebration ---
                st.balloons()  # Fun celebration animation
                
                # --- Success Header ---
                st.markdown(
                    """
                    <div style="background: linear-gradient(90deg, #56ab2f, #a8e6cf); padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
                        <h2 style="color: white; margin: 0;">🎉 Pipeline Execution Complete!</h2>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Your AutoPilotML model is ready!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # --- Results Summary Cards ---
                # Extract key results from the pipeline output
                rating = results['overall_rating']
                # Determine color based on rating score
                rating_color = "#ff6b6b" if rating < 5 else "#feca57" if rating < 8 else "#4ecdc4"
                # Generate description based on rating
                rating_description = (
                    "Poor performance - may need more data or feature engineering" if rating < 5 
                    else "Good performance - model is reliable for most use cases" if rating < 8 
                    else "Excellent performance - model is highly accurate and ready for production"
                )

                # Display three key metrics in card format
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
                        <div class="metric-card tooltip" style="flex: 1; min-width: 250px; max-width: 350px; text-align: center;">
                            <h3 style="color: #4ecdc4; margin: 0;">🎯 Task Type</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{results['task'].capitalize()}</h2>
                            <span class="tooltiptext">
                                <strong>Task Type:</strong> Indicates whether this is a classification problem 
                                (predicting categories) or regression problem (predicting continuous values). 
                                AutoPilotML automatically detects the appropriate task type based on your target variable.
                            </span>
                        </div>
                        <div class="metric-card tooltip" style="flex: 1; min-width: 250px; max-width: 350px; text-align: center;">
                            <h3 style="color: #ff6b6b; margin: 0;">🏆 Best Model</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{results['best_model']}</h2>
                            <span class="tooltiptext">
                                <strong>Best Model:</strong> The machine learning algorithm that performed best 
                                on your dataset. AutoPilotML tested 12+ different algorithms and selected this one 
                                based on cross-validation performance and your chosen strategy.
                            </span>
                        </div>
                        <div class="metric-card tooltip" style="flex: 1; min-width: 250px; max-width: 350px; text-align: center;">
                            <h3 style="color: {rating_color}; margin: 0;">⭐ Overall Rating</h3>
                            <h2 style="color: white; margin: 0.5rem 0 0 0;">{rating:.1f}/10</h2>
                            <span class="tooltiptext">
                                <strong>Overall Rating:</strong> A comprehensive score (1-10) based on model accuracy, 
                                cross-validation stability, and data quality. <br><br>
                                <strong>Your Score:</strong> {rating_description}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --- Performance Metrics Display ---
                # Shows detailed performance metrics in two columns
                st.subheader("📊 Model Performance")
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.write("**Performance Metrics:**")
                    # Loop through all metrics and display them
                    for key, value in results['metrics'].items():
                        if isinstance(value, float):
                            # Format floating point numbers to 4 decimal places
                            st.write(f"• **{key.upper()}**: {value:.4f}")
                        else:
                            # Display non-numeric values as-is
                            st.write(f"• **{key.upper()}**: {value}")

                with metrics_col2:
                    st.write("**Model Summary:**")
                    # Display text summary of the best model
                    st.write(results['summary'])

                # --- Developer Logs Section ---
                # Collapsible section showing detailed technical logs
                with st.expander("👨‍💻 For Developers", expanded=False):
                    # Display each log entry on a separate line
                    for log in results['logs']:
                        st.text(log)

                # --- Visualization Display ---
                # Shows plots generated by the pipeline in organized tabs
                if results['plots']:
                    st.subheader("📈 Visualizations")
                    
                    # Create tab labels for each plot
                    num_plots = len(results['plots'])
                    tab_labels = []
                    for i in range(num_plots):
                        tab_labels.append(f"Plot {i+1:02d}")  # Zero-padded numbers for alignment
                    
                    # Create tabs for each plot
                    plot_tabs = st.tabs(tab_labels)
                    
                    # Display each plot in its own tab
                    for i, (tab, plot) in enumerate(zip(plot_tabs, results['plots'])):
                        with tab:
                            plot_container = st.container()
                            with plot_container:
                                # Set consistent figure size for all plots
                                if hasattr(plot, 'set_size_inches'):
                                    plot.set_size_inches(6, 4)  # Width=6, Height=4 inches
                                
                                # Display the matplotlib figure
                                st.pyplot(plot, use_container_width=False)

                # --- SHAP Feature Importance Analysis ---
                # SHAP (SHapley Additive exPlanations) shows which features matter most for predictions
                if results['shap_analysis'] and results['shap_analysis'].get('feature_importance') is not None:
                    st.subheader("🔍 SHAP Feature Importance Analysis")
                    
                    shap_col1, shap_col2 = st.columns(2)
                    
                    with shap_col1:
                        st.write("**Top Most Important Features:**")
                        # Extract feature importance data
                        feature_importance = results['shap_analysis']['feature_importance']
                        feature_names = results['shap_analysis'].get('feature_names', 
                                                                   [f"feature_{i}" for i in range(len(feature_importance))])
                        
                        # Find the top 10 most important features
                        top_indices = np.argsort(feature_importance)[-10:][::-1]  # Sort descending, take top 10
                        top_features = [feature_names[i] for i in top_indices]
                        top_importance = [feature_importance[i] for i in top_indices]
                        
                        # Create DataFrame for the bar chart
                        importance_df = pd.DataFrame({
                            'Feature': top_features,
                            'Importance': top_importance
                        })
                        st.bar_chart(importance_df.set_index('Feature'))
                    
                    with shap_col2:
                        st.write("**SHAP Analysis Summary:**")
                        # Display analysis logs if available
                        for log in results['shap_analysis'].get('analysis_logs', []):
                            st.text(log)

                # --- Model Download Section ---
                # Allows users to download the trained model for use in other applications
                st.subheader("💾 Download Trained Model")
                
                # Check if model file exists
                if os.path.exists(results['model_path']):
                    # Read the model file and create download button
                    with open(results['model_path'], "rb") as f:
                        st.download_button(
                            label="⬇️ Download Best Model (.pkl)",
                            data=f.read(),  # File contents
                            file_name=f"{results['best_model'].replace(' ', '_')}_model.pkl",  # Clean filename
                            mime="application/octet-stream"  # Binary file type
                        )
                    st.success(f"✅ Model saved at: {results['model_path']}")
                else:
                    st.error("❌ Model file not found")

                # --- Model Comparison Chart ---
                # Shows performance of all tested models for comparison
                if results['model_scores']:
                    st.subheader("🏅 All Model Scores Comparison")
                    # Convert model scores dictionary to DataFrame
                    scores_df = pd.DataFrame(list(results['model_scores'].items()), 
                                           columns=['Model', 'Score'])
                    # Sort by score (highest first)
                    scores_df = scores_df.sort_values('Score', ascending=False)
                    # Display as horizontal bar chart
                    st.bar_chart(scores_df.set_index('Model'))

            except Exception as e:
                # Error handling - shows detailed error information if pipeline fails
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                st.error("Please check your dataset and try again.")
                st.exception(e)  # Shows full error traceback for debugging

# =============================
#   Case 2: No Dataset Uploaded
# =============================
# This section displays welcome content and instructions when no file is uploaded
else:
    # --- Welcome Message ---
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
    
    # --- Features Showcase ---
    # Three-column layout showing what AutoPilotML can do
    st.markdown("### 🚀 What AutoPilotML Can Do")
    
    col1, col2, col3 = st.columns(3)  # Three equal columns
    
    with col1:
        # Smart Automation features
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
        # Model Training features
        st.markdown(
            """
            <div class="metric-card" style="text-align: center;">
                <h3 style="color: #ff6b6b; margin: 0 0 1rem 0;">🏆 Model Training</h3>
                <ul style="color: rgba(255,255,255,0.8); text-align: left; padding-left: 1rem;">
                    <li>12+ ML algorithms</li>
                    <li>Hyperparameter tuning</li>
                    <li>Cross-validation and GridSearchCV </li>
                    <li>Performance optimization</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        # Analysis & Export features
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
    
    # --- Getting Started Instructions ---
    # Step-by-step guide for new users
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
    
    # --- Sample Datasets Section ---
    # Shows available local datasets as clickable cards
    st.markdown("### 📂 Sample Datasets Available")
    
    # Define local datasets with their file paths and display information
    sample_datasets = [
        {"name": "❤️ Heart Disease", "description": "Heart Disease prediction", "file": "heart.csv", "target_hint": "target"},
        {"name": "🧬 Life Insurance", "description": "Insurance claim prediction", "file": "life_insurance.csv", "target_hint": "claim"},
        {"name": "🧊🚢 Titanic", "description": "Survival prediction", "file": "titanic.csv", "target_hint": "Survived"},
        {"name": "🏦 Bank Marketing", "description": "Campaign success prediction", "file": "bank.csv", "target_hint": "y"},
        {"name": "🌸 Iris", "description": "Classic flower classification", "file": "iris.csv", "target_hint": "species"},
        {"name": "🏠 Housing", "description": "Price prediction", "file": "housing.csv", "target_hint": "median_house_value"},
    ]
    
    # Initialize session state for selected dataset
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'loaded_df' not in st.session_state:
        st.session_state.loaded_df = None
    if 'target_hint' not in st.session_state:
        st.session_state.target_hint = None
    
    # Display datasets in a 2-column grid layout with styled cards
    for i in range(0, len(sample_datasets), 2):  # Process 2 datasets at a time
        col1, col2 = st.columns(2)
        
        # Left column dataset
        if i < len(sample_datasets):
            with col1:
                dataset = sample_datasets[i]
                # Get dataset info first to display in card
                dataset_path = f"Datasets/{dataset['file']}"
                try:
                    if os.path.exists(dataset_path):
                        temp_df = pd.read_csv(dataset_path)
                        rows = temp_df.shape[0]
                        features = temp_df.shape[1]
                        
                        # Create styled card with dataset information
                        card_html = f"""
                        <div class="dataset-card" style="
                            border-left: 4px solid #4ecdc4; 
                            background: rgba(78, 205, 196, 0.1); 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin-bottom: 0.8rem;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onclick="document.getElementById('load_btn_{i}').click();">
                            <h3 style="color: white; margin: 0 0 0.3rem 0; font-size: 1.1rem;">{dataset['name']}</h3>
                            <p style="color: rgba(255,255,255,0.8); margin: 0 0 0.5rem 0; font-size: 0.85rem;">{dataset['description']}</p>
                            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">{features} features • {rows} rows</p>
                            <p style="color: #feca57; margin: 0.3rem 0 0 0; font-size: 0.8rem;">💡 Target: {dataset['target_hint']}</p>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Full width load button that gets triggered by card click
                        if st.button("Load Dataset", key=f"load_btn_{i}", help=f"Load {dataset['name']} dataset", use_container_width=True):
                            st.session_state.loaded_df = temp_df
                            st.session_state.selected_dataset = dataset['name']
                            st.session_state.target_hint = dataset['target_hint']
                            # Set the loaded dataset as if it was uploaded
                            st.session_state.dataset_loaded_from_sample = True
                            st.success(f"✅ {dataset['name']} dataset loaded successfully!")
                            st.rerun()
                    else:
                        st.error(f"❌ Dataset file not found: {dataset_path}")
                except Exception as e:
                    st.error(f"❌ Error loading dataset info: {str(e)}")
        
        # Right column dataset
        if i + 1 < len(sample_datasets):
            with col2:
                dataset = sample_datasets[i + 1]
                # Get dataset info first to display in card
                dataset_path = f"Datasets/{dataset['file']}"
                try:
                    if os.path.exists(dataset_path):
                        temp_df = pd.read_csv(dataset_path)
                        rows = temp_df.shape[0]
                        features = temp_df.shape[1]
                        
                        # Determine card color based on index
                        colors = ["#ff6b6b", "#4ecdc4", "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd"]
                        color = colors[(i + 1) % len(colors)]
                        
                        # Create styled card with dataset information
                        card_html = f"""
                        <div class="dataset-card" style="
                            border-left: 4px solid {color}; 
                            background: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1); 
                            padding: 1rem; 
                            border-radius: 8px; 
                            margin-bottom: 0.8rem;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        " onclick="document.getElementById('load_btn_{i+1}').click();">
                            <h3 style="color: white; margin: 0 0 0.3rem 0; font-size: 1.1rem;">{dataset['name']}</h3>
                            <p style="color: rgba(255,255,255,0.8); margin: 0 0 0.5rem 0; font-size: 0.85rem;">{dataset['description']}</p>
                            <p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 0.8rem;">{features} features • {rows} rows</p>
                            <p style="color: #feca57; margin: 0.3rem 0 0 0; font-size: 0.8rem;">💡 Target: {dataset['target_hint']}</p>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                        # Full width load button that gets triggered by card click
                        if st.button("Load Dataset", key=f"load_btn_{i+1}", help=f"Load {dataset['name']} dataset", use_container_width=True):
                            st.session_state.loaded_df = temp_df
                            st.session_state.selected_dataset = dataset['name']
                            st.session_state.target_hint = dataset['target_hint']
                            # Set the loaded dataset as if it was uploaded
                            st.session_state.dataset_loaded_from_sample = True
                            st.success(f"✅ {dataset['name']} dataset loaded successfully!")
                            st.rerun()
                    else:
                        st.error(f"❌ Dataset file not found: {dataset_path}")
                except Exception as e:
                    st.error(f"❌ Error loading dataset info: {str(e)}")
    
    # Pro tip for users
    st.info("💡 **Pro Tip**: Click on any dataset card above to instantly load and explore the data!")
