"""
Streamlit Deployment App for Credit Card Fraud Detection

This module provides an interactive web application for predicting
credit card fraud using trained models.

To run: streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str):
    """Load a trained model from disk."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_scaler(scaler_path: str):
    """Load a fitted scaler from disk."""
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


def get_feature_inputs():
    """Get input features from user."""
    st.sidebar.header("Transaction Details")
    
    # Time feature
    time = st.sidebar.slider(
        "Time (seconds since midnight)",
        min_value=0,
        max_value=86400,
        value=43200,
        step=60,
        help="Time elapsed in seconds since the first transaction in the dataset"
    )
    
    # Amount feature
    amount = st.sidebar.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        max_value=25000.0,
        value=100.0,
        step=0.01,
        help="The transaction amount in dollars"
    )
    
    # PCA features (V1-V28)
    st.sidebar.subheader("Anonymized Features (V1-V28)")
    
    v_features = {}
    
    # Use expanders to organize the many features
    with st.sidebar.expander("Features V1-V7", expanded=False):
        for i in range(1, 8):
            v_features[f'V{i}'] = st.number_input(
                f"V{i}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key=f'v{i}'
            )
    
    with st.sidebar.expander("Features V8-V14", expanded=False):
        for i in range(8, 15):
            v_features[f'V{i}'] = st.number_input(
                f"V{i}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key=f'v{i}'
            )
    
    with st.sidebar.expander("Features V15-V21", expanded=False):
        for i in range(15, 22):
            v_features[f'V{i}'] = st.number_input(
                f"V{i}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key=f'v{i}'
            )
    
    with st.sidebar.expander("Features V22-V28", expanded=False):
        for i in range(22, 29):
            v_features[f'V{i}'] = st.number_input(
                f"V{i}",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                key=f'v{i}'
            )
    
    # Create feature dictionary
    features = {'Time': time, 'Amount': amount}
    features.update(v_features)
    
    return features


def create_feature_dataframe(features: dict) -> pd.DataFrame:
    """Convert feature dictionary to DataFrame."""
    df = pd.DataFrame([features])
    
    # Ensure columns are in correct order
    expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Add missing columns with default value 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[expected_cols]
    
    return df


def predict_fraud(model, scaler, features_df: pd.DataFrame) -> tuple:
    """Make fraud prediction."""
    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = scaler.transform(features_df)
    else:
        features_scaled = features_df.values
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features_scaled)[0][1]
    else:
        proba = 0.5
    
    return prediction, proba


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">💳 Credit Card Fraud Detection</p>', unsafe_allow_html=True)
    st.markdown("""
        This application uses machine learning to detect potentially fraudulent credit card transactions.
        Input the transaction details below to get a real-time fraud prediction.
    """)
    
    # Sidebar for model selection and inputs
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_dir = "models"
    available_models = []
    
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib') or f.endswith('.joblib')]
        available_models = [f.replace('_model.joblib', '').replace('.joblib', '') for f in model_files]
    
    if available_models:
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            options=available_models,
            index=0 if 'XGBoost' not in available_models else available_models.index('XGBoost') if 'XGBoost' in available_models else 0
        )
        model_path = os.path.join(model_dir, f"{selected_model_name}_model.joblib")
        
        # Try alternate naming
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, f"{selected_model_name}.joblib")
    else:
        st.sidebar.warning("No trained models found in the 'models' directory.")
        selected_model_name = "Default"
        model_path = None
    
    # Scaler selection
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    # Load model and scaler
    model = None
    scaler = None
    
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)
        st.sidebar.success(f"✅ Model '{selected_model_name}' loaded successfully!")
    else:
        st.sidebar.error("❌ Could not load model. Please train a model first.")
    
    if os.path.exists(scaler_path):
        scaler = load_scaler(scaler_path)
        st.sidebar.success("✅ Scaler loaded successfully!")
    
    # Display model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    if model:
        st.sidebar.write(f"**Type:** {type(model).__name__}")
        st.sidebar.write(f"**Features:** 30 (Time, Amount, V1-V28)")
    
    # Get user inputs
    features = get_feature_inputs()
    features_df = create_feature_dataframe(features)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Input Summary")
        
        # Display input features in a nice format
        input_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Value': [f"{v:.2f}" if isinstance(v, float) else str(v) for v in features.values()]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🎯 Prediction")
        
        if model is not None:
            # Make prediction
            prediction, probability = predict_fraud(model, scaler, features_df)
            
            # Display result with appropriate styling
            if prediction == 1:
                st.markdown("""
                    <div class="danger-box">
                        <h3 style="color: #dc3545; margin: 0;">⚠️ FRAUD DETECTED</h3>
                        <p style="margin: 0.5rem 0 0 0;">This transaction shows signs of potential fraud.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Risk level based on probability
                if probability > 0.9:
                    risk_level = "🔴 Very High"
                elif probability > 0.7:
                    risk_level = "🟠 High"
                elif probability > 0.5:
                    risk_level = "🟡 Medium"
                else:
                    risk_level = "🟢 Low"
                
                st.metric("Fraud Probability", f"{probability:.2%}", delta=risk_level)
                
            else:
                st.markdown("""
                    <div class="success-box">
                        <h3 style="color: #28a745; margin: 0;">✅ LEGITIMATE</h3>
                        <p style="margin: 0.5rem 0 0 0;">This transaction appears to be normal.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                legitimacy = 1 - probability
                st.metric("Legitimacy Probability", f"{legitimacy:.2%}")
            
            # Additional metrics
            st.markdown("---")
            st.write("**Confidence Metrics:**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fraud Score", f"{probability:.2%}")
            with col_b:
                st.metric("Decision Threshold", "50%")
        
        else:
            st.warning("⚠️ Please load a model to make predictions.")
    
    # Additional information section
    st.markdown("---")
    st.subheader("ℹ️ About This Application")
    
    with st.expander("How It Works", expanded=False):
        st.write("""
            This fraud detection system uses advanced machine learning algorithms trained on 
            historical credit card transaction data. The model analyzes various features of 
            each transaction to determine the likelihood of fraud.
            
            **Key Features:**
            - **Time**: Time elapsed since the first transaction
            - **Amount**: Transaction amount in dollars
            - **V1-V28**: Anonymized principal components from the original features
            
            **Model Performance:**
            The trained models are evaluated using metrics such as:
            - ROC-AUC Score
            - Precision and Recall
            - F1-Score
            - Matthews Correlation Coefficient
        """)
    
    with st.expander("Important Notes", expanded=False):
        st.write("""
            ⚠️ **Disclaimer:** This is a demonstration application. In production environments:
            
            1. Models should be regularly retrained with new data
            2. Predictions should be reviewed by human analysts
            3. False positives and false negatives have real-world consequences
            4. Additional features and domain knowledge should be incorporated
            5. Proper monitoring and logging should be implemented
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>Built with Streamlit | Machine Learning Pipeline for Credit Card Fraud Detection</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
