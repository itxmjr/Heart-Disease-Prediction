"""
Streamlit Web Application for Heart Disease Prediction
======================================================

This creates an interactive web interface where users can:
1. Input their health data
2. Get heart disease risk prediction
3. See probability and risk factors
4. Explore the data and model performance

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

# Add src to path
sys.path.append('src')
from src.dataloader import load_data as loader_load_data

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #e74c3c;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #2ecc71;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)


# ========================================
# LOAD MODEL AND DATA
# ========================================

@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        model_data = joblib.load('models/heart_model.pkl')
        return model_data
    except FileNotFoundError:
        return None

@st.cache_data
def load_data():
    """Load the dataset for exploration."""
    # We use the loader from src, passing the expected path
    # Assuming standard path data/heart.csv
    return loader_load_data('data/heart.csv')


# ========================================
# SIDEBAR
# ========================================

def create_sidebar():
    """Create the sidebar with navigation and info."""
    st.sidebar.markdown("# 🫀 Navigation")
    
    page = st.sidebar.radio(
        "Go to:",
        ["🏠 Home", "🔮 Prediction", "📊 Data Explorer", "📈 Model Performance", "ℹ️ About"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Quick Info")
    st.sidebar.info("""
        This app predicts heart disease risk based on 13 health parameters.
        
        **Dataset:** UCI Heart Disease Dataset
        
        **Model:** Machine Learning Classification
    """)
    
    return page


# ========================================
# HOME PAGE
# ========================================

def show_home_page():
    """Display the home page."""
    st.markdown('<h1 class="main-header">🫀 Heart Disease Prediction System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Heart Health Risk Assessment</p>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 Accurate</h3>
                <p>Built using advanced ML algorithms with high accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>⚡ Fast</h3>
                <p>Get instant predictions based on your health data</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>🔒 Secure</h3>
                <p>Your data is processed locally and not stored</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔍 What This App Does")
    st.markdown("""
        This application uses machine learning to predict the likelihood of heart disease 
        based on various health parameters. It analyzes:
        
        - **Demographic factors** (age, sex)
        - **Vital signs** (blood pressure, heart rate)
        - **Lab results** (cholesterol, blood sugar)
        - **Clinical features** (chest pain type, ECG results)
        
        > ⚠️ **Disclaimer:** This tool is for educational purposes only and should not 
        replace professional medical advice.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        df = load_data()
        if df is not None:
            st.write(f"- **Total Patients:** {len(df)}")
            st.write(f"- **Features:** {len(df.columns) - 1}")
            disease_pct = df['target'].mean() * 100
            st.write(f"- **Heart Disease Cases:** {disease_pct:.1f}%")
    
    with col2:
        st.markdown("### 🤖 Model Info")
        model_data = load_model()
        if model_data:
            st.write(f"- **Model Type:** {model_data['model_name']}")
            st.write(f"- **Features Used:** {len(model_data['feature_names'])}")


# ========================================
# PREDICTION PAGE
# ========================================

def show_prediction_page():
    """Display the prediction page with input form."""
    st.markdown('<h1 class="main-header">🔮 Heart Disease Risk Prediction</h1>', 
                unsafe_allow_html=True)
    
    model_data = load_model()
    
    if model_data is None:
        st.error("⚠️ Model not found! Please train the model first by running the ML pipeline.")
        st.code("python -c \"from src.model import *; from src.data_preprocessing import *; df = load_data('data/heart.csv'); run_complete_ml_pipeline(df)\"")
        return
    
    st.markdown("### 📝 Enter Patient Information")
    st.markdown("Fill in the health parameters below to get a prediction.")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", min_value=20, max_value=100, value=50, 
                                 help="Patient's age in years")
            
            sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], 
                              format_func=lambda x: x[0], 
                              help="Biological sex")
            
            cp = st.selectbox("Chest Pain Type", 
                             options=[(0, "Typical Angina"), (1, "Atypical Angina"), 
                                     (2, "Non-anginal Pain"), (3, "Asymptomatic")],
                             format_func=lambda x: x[1],
                             help="Type of chest pain experienced")
            
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                       min_value=80, max_value=200, value=120,
                                       help="Blood pressure at rest")
            
            chol = st.number_input("Serum Cholesterol (mg/dl)", 
                                   min_value=100, max_value=600, value=200,
                                   help="Total cholesterol level")
        
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                              options=[(0, "No"), (1, "Yes")],
                              format_func=lambda x: x[1],
                              help="Is fasting blood sugar > 120 mg/dl?")
            
            restecg = st.selectbox("Resting ECG Results",
                                   options=[(0, "Normal"), (1, "ST-T Abnormality"), 
                                           (2, "Left Ventricular Hypertrophy")],
                                   format_func=lambda x: x[1],
                                   help="Resting electrocardiographic results")
            
            thalach = st.number_input("Maximum Heart Rate", 
                                      min_value=60, max_value=220, value=150,
                                      help="Maximum heart rate achieved during exercise")
            
            exang = st.selectbox("Exercise Induced Angina",
                                options=[(0, "No"), (1, "Yes")],
                                format_func=lambda x: x[1],
                                help="Does exercise induce angina?")
        
        with col3:
            oldpeak = st.number_input("ST Depression (oldpeak)", 
                                      min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                      help="ST depression induced by exercise relative to rest")
            
            slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                options=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
                                format_func=lambda x: x[1],
                                help="Slope of the peak exercise ST segment")
            
            ca = st.selectbox("Number of Major Vessels (0-3)",
                             options=[0, 1, 2, 3],
                             help="Number of major vessels colored by fluoroscopy")
            
            thal = st.selectbox("Thalassemia",
                               options=[(0, "Normal"), (1, "Fixed Defect"), 
                                       (2, "Reversible Defect"), (3, "Unknown")],
                               format_func=lambda x: x[1],
                               help="Thalassemia blood disorder type")
        
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex[1],
            'cp': cp[0],
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs[0],
            'restecg': restecg[0],
            'thalach': thalach,
            'exang': exang[0],
            'oldpeak': oldpeak,
            'slope': slope[0],
            'ca': ca,
            'thal': thal[0]
        }
        
        # Make prediction
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        
        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct order
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h2 style="color: #e74c3c;">⚠️ HIGH RISK</h2>
                        <p style="font-size: 1.2rem;">Heart Disease Risk Detected</p>
                        <h1 style="color: #e74c3c;">{probability*100:.1f}%</h1>
                        <p>Probability of Heart Disease</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h2 style="color: #2ecc71;">✅ LOW RISK</h2>
                        <p style="font-size: 1.2rem;">No Significant Heart Disease Risk</p>
                        <h1 style="color: #2ecc71;">{(1-probability)*100:.1f}%</h1>
                        <p>Probability of Being Healthy</p>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Risk gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if probability > 0.5 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},
                        {'range': [30, 60], 'color': "#fff3cd"},
                        {'range': [60, 100], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        if prediction == 1:
            st.warning("""
                **Based on the analysis, we recommend:**
                - Schedule an appointment with a cardiologist
                - Get comprehensive heart health screening
                - Monitor blood pressure regularly
                - Consider lifestyle modifications (diet, exercise)
                - Reduce stress and quit smoking if applicable
            """)
        else:
            st.success("""
                **To maintain good heart health:**
                - Continue regular exercise (30 mins/day)
                - Maintain a heart-healthy diet
                - Keep cholesterol and blood pressure in check
                - Schedule regular health checkups
                - Manage stress effectively
            """)
        
        st.info("⚠️ **Important:** This is a screening tool only. Please consult a healthcare professional for proper diagnosis and treatment.")


# ========================================
# DATA EXPLORER PAGE
# ========================================

def show_data_explorer():
    """Display the data exploration page."""
    st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
    
    df = load_data()
    
    if df is None:
        st.error("Dataset not found! Please place heart.csv in the data/ folder.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Visualizations", "🔗 Correlations"])
    
    with tab1:
        st.markdown("### Raw Dataset")
        st.dataframe(df.head(50), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.markdown("### Distribution of Features")
        
        feature = st.selectbox("Select feature to visualize:", df.columns.tolist())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=feature, color='target', 
                              title=f'Distribution of {feature}',
                              color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                              labels={'target': 'Heart Disease'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='target', y=feature, color='target',
                        title=f'{feature} by Heart Disease Status',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                        labels={'target': 'Heart Disease'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Target distribution
        st.markdown("### Target Variable Distribution")
        fig = px.pie(df, names='target', title='Heart Disease Distribution',
                    color='target', color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Correlation Heatmap")
        
        corr = df.corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Matrix')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top Correlations with Target")
        target_corr = corr['target'].drop('target').sort_values(key=abs, ascending=False)
        
        fig = px.bar(x=target_corr.values, y=target_corr.index,
                    orientation='h', title='Feature Correlation with Heart Disease',
                    color=target_corr.values, color_continuous_scale='RdBu_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ========================================
# MODEL PERFORMANCE PAGE
# ========================================

def show_model_performance():
    """Display model performance metrics."""
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    model_data = load_model()
    
    if model_data is None:
        st.error("Model not found! Please train the model first.")
        return
    
    st.markdown(f"### Current Model: **{model_data['model_name']}**")
    
    # Load saved images if available
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        if os.path.exists('outputs/confusion_matrix.png'):
            st.image('outputs/confusion_matrix.png')
        else:
            st.info("Run the ML pipeline to generate confusion matrix visualization.")
    
    with col2:
        st.markdown("### 📈 ROC Curves")
        if os.path.exists('outputs/roc_curves.png'):
            st.image('outputs/roc_curves.png')
        else:
            st.info("Run the ML pipeline to generate ROC curve visualization.")
    
    st.markdown("### 🎯 Feature Importance")
    if os.path.exists('outputs/feature_importance.png'):
        st.image('outputs/feature_importance.png')
    else:
        st.info("Run the ML pipeline to generate feature importance visualization.")
    
    st.markdown("### 📚 Understanding the Metrics")
    
    with st.expander("What is Accuracy?"):
        st.markdown("""
            **Accuracy** measures the proportion of correct predictions.
            
            Formula: `(True Positives + True Negatives) / Total Predictions`
            
            - 100% = Perfect predictions
            - 50% = Random guessing (for balanced data)
        """)
    
    with st.expander("What is ROC-AUC?"):
        st.markdown("""
            **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve) measures 
            how well the model distinguishes between classes.
            
            - 1.0 = Perfect separation
            - 0.5 = No separation (random)
            
            It's useful because it's independent of the classification threshold.
        """)
    
    with st.expander("What is the Confusion Matrix?"):
        st.markdown("""
            The **Confusion Matrix** shows:
            
            - **True Positives (TP)**: Correctly predicted disease
            - **True Negatives (TN)**: Correctly predicted no disease
            - **False Positives (FP)**: Predicted disease, but actually healthy
            - **False Negatives (FN)**: Predicted healthy, but actually has disease
            
            In medical applications, we especially care about minimizing False Negatives!
        """)


# ========================================
# ABOUT PAGE
# ========================================

def show_about_page():
    """Display the about page."""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        ## 🎯 Project Overview
        
        This Heart Disease Prediction System was built as a learning project to understand:
        
        - **Data Preprocessing**: Cleaning and preparing medical data
        - **Exploratory Data Analysis**: Understanding patterns and relationships
        - **Machine Learning**: Building classification models
        - **Model Evaluation**: Using proper metrics for medical predictions
        - **Web Development**: Creating interactive interfaces with Streamlit
        
        ## 📊 Dataset
        
        The [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) 
        contains 14 attributes collected from patients, including:
        
        | Feature | Description |
        |---------|-------------|
        | age | Age in years |
        | sex | Sex (1 = male, 0 = female) |
        | cp | Chest pain type (0-3) |
        | trestbps | Resting blood pressure |
        | chol | Serum cholesterol |
        | fbs | Fasting blood sugar > 120 mg/dl |
        | restecg | Resting ECG results |
        | thalach | Maximum heart rate achieved |
        | exang | Exercise induced angina |
        | oldpeak | ST depression |
        | slope | Slope of peak exercise ST segment |
        | ca | Number of major vessels (0-3) |
        | thal | Thalassemia |
        
        ## 🤖 Machine Learning Models
        
        We trained and compared multiple models:
        
        1. **Logistic Regression**: A linear model great for interpretability
        2. **Decision Tree**: A tree-based model that's easy to visualize
        3. **Random Forest**: An ensemble of decision trees for better accuracy
        
        ## 🛠️ Technologies Used
        
        - **Python**: Core programming language
        - **Pandas & NumPy**: Data manipulation
        - **Scikit-learn**: Machine learning
        - **Matplotlib & Seaborn**: Static visualizations
        - **Plotly**: Interactive visualizations
        - **Streamlit**: Web application framework
        
        ## ⚠️ Disclaimer
        
        This application is for **educational purposes only**. It should NOT be used as a 
        substitute for professional medical advice, diagnosis, or treatment. Always seek 
        the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.
        
        ## 👨‍💻 Author
        
        Built as a learning project for understanding ML in healthcare.
        
        ---
        
        *Made with ❤️ using Python and Streamlit*
    """)


# ========================================
# MAIN APP
# ========================================

def main():
    """Main application function."""
    page = create_sidebar()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Prediction":
        show_prediction_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "📈 Model Performance":
        show_model_performance()
    elif page == "ℹ️ About":
        show_about_page()


if __name__ == "__main__":
    main()