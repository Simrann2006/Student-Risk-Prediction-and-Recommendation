# STUDENT RISK PREDICTION SYSTEM - STREAMLIT UI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))

from student_risk_prediction_system import load_dataset, train_all_models, FEATURES

st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1e40af; text-align: center; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 2rem; }
    .risk-high { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); }
    .risk-low { background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4); }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; margin: 0.5rem 0; }
    .info-box { background: #f0f9ff; border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.75rem 2rem; font-size: 1.1rem; font-weight: bold; border-radius: 10px; }
    .stButton>button:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return load_dataset("dataset/Portuguese_cleaned.csv")

@st.cache_resource
def train_models(df):
    return train_all_models(df, silent=True)


# Load data and train models
df = load_data()
models = train_models(df)

# Sidebar navigation
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Select Page:", ["🎯 Predict Risk", "📚 Recommendations", "📊 Model Performance", "ℹ️ About"])


def show_predict_page():
    st.markdown('<p class="main-title">🎯 Student Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter student information to predict academic risk</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        g1 = st.slider("First Period Grade (G1)", 0, 20, 10)
        g2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
        
    with col2:
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], format_func=lambda x: {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[x])
        failures = st.selectbox("Past Failures", [0, 1, 2, 3])
        absences = st.slider("Absences", 0, 93, 5)
        
    with col3:
        schoolsup = st.selectbox("School Support", ["no", "yes"])
        famsup = st.selectbox("Family Support", ["no", "yes"])
        paid = st.selectbox("Paid Classes", ["no", "yes"])
        internet = st.selectbox("Internet Access", ["no", "yes"])
        higher = st.selectbox("Wants Higher Education", ["no", "yes"])
        health = st.slider("Health (1-5)", 1, 5, 3)
        goout = st.slider("Going Out (1-5)", 1, 5, 3)
    
    if st.button("🔮 Predict Risk", use_container_width=True):
        input_data = {
            'G1': g1, 'G2': g2, 'studytime': studytime, 'failures': failures, 'absences': absences,
            'schoolsup': 1 if schoolsup == "yes" else 0, 'famsup': 1 if famsup == "yes" else 0,
            'paid': 1 if paid == "yes" else 0, 'internet': 1 if internet == "yes" else 0,
            'higher': 1 if higher == "yes" else 0, 'health': health, 'goout': goout
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = models['scaler'].transform(input_df)
        
        lr_pred = models['lr_model'].predict(input_scaled)[0]
        rf_pred = models['rf_model'].predict(input_scaled)[0]
        rf_proba = models['rf_model'].predict_proba(input_scaled)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression")
            if lr_pred == 1:
                st.markdown('<div class="risk-high">⚠️ AT RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ NOT AT RISK</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Random Forest")
            if rf_pred == 1:
                st.markdown('<div class="risk-high">⚠️ AT RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ NOT AT RISK</div>', unsafe_allow_html=True)
        
        st.subheader("Risk Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=rf_proba[1] * 100, title={'text': "Risk Level (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ef4444" if rf_proba[1] > 0.5 else "#10b981"}}
        ))
        st.plotly_chart(fig, use_container_width=True)


def show_recommendations_page():
    st.markdown('<p class="main-title">📚 KNN Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Find similar students and get personalized learning advice</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        g1 = st.slider("G1", 0, 20, 8, key="r_g1")
        g2 = st.slider("G2", 0, 20, 8, key="r_g2")
        studytime = st.selectbox("Study Time", [1, 2, 3, 4], key="r_st")
        failures = st.selectbox("Failures", [0, 1, 2, 3], key="r_f")
        absences = st.slider("Absences", 0, 50, 5, key="r_abs")
        health = st.slider("Health", 1, 5, 3, key="r_h")
        
    with col2:
        goout = st.slider("Going Out", 1, 5, 3, key="r_go")
        schoolsup = st.selectbox("School Support", ["no", "yes"], key="r_ss")
        famsup = st.selectbox("Family Support", ["no", "yes"], key="r_fs")
        paid = st.selectbox("Paid Classes", ["no", "yes"], key="r_p")
        internet = st.selectbox("Internet", ["no", "yes"], key="r_i")
        higher = st.selectbox("Higher Education", ["no", "yes"], key="r_hi")
    
    if st.button("🎯 Get Recommendations", use_container_width=True):
        input_data = {
            'G1': g1, 'G2': g2, 'studytime': studytime, 'failures': failures, 'absences': absences,
            'schoolsup': 1 if schoolsup == "yes" else 0, 'famsup': 1 if famsup == "yes" else 0,
            'paid': 1 if paid == "yes" else 0, 'internet': 1 if internet == "yes" else 0,
            'higher': 1 if higher == "yes" else 0, 'health': health, 'goout': goout
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = models['scaler'].transform(input_df)
        
        distances, indices = models['knn'].kneighbors(input_scaled)
        similar_outcomes = models['y_train'].iloc[indices[0]]
        success_rate = (similar_outcomes == 0).mean() * 100
        
        st.success(f"Similar students success rate: {success_rate:.1f}%")
        
        st.subheader("Recommendations")
        if studytime < 3:
            st.info("📚 Increase study time to 5-10 hours per week")
        if absences > 10:
            st.info("📅 Reduce absences - attend classes regularly")
        if failures > 0:
            st.info("🎯 Get extra tutoring to address past difficulties")
        if schoolsup == "no":
            st.info("🏫 Seek school support programs")
        if goout > 3:
            st.info("⚖️ Balance social activities with study time")


def show_performance_page():
    st.markdown('<p class="main-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Compare model accuracy and analyze feature importance</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LR Test Accuracy", f"{models['lr_accuracy']*100:.1f}%", delta="Test")
    col2.metric("RF Test Accuracy", f"{models['rf_accuracy']*100:.1f}%", delta="Test")
    col3.metric("LR Validation", f"{models['lr_val_accuracy']*100:.1f}%", delta="Val")
    col4.metric("RF Validation", f"{models['rf_val_accuracy']*100:.1f}%", delta="Val")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Logistic Regression Confusion Matrix")
        fig = px.imshow(models['lr_cm'], text_auto=True, color_continuous_scale='Blues',
                       x=['Not At Risk', 'At Risk'], y=['Not At Risk', 'At Risk'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌲 Random Forest Confusion Matrix")
        fig = px.imshow(models['rf_cm'], text_auto=True, color_continuous_scale='Greens',
                       x=['Not At Risk', 'At Risk'], y=['Not At Risk', 'At Risk'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Feature Importance (Random Forest)")
    importance = pd.DataFrame({'Feature': FEATURES, 'Importance': models['rf_model'].feature_importances_})
    importance = importance.sort_values('Importance', ascending=True)
    
    fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Viridis',
                 title='Which features matter most for prediction?')
    fig.update_layout(height=450, showlegend=False, coloraxis_showscale=True)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    st.markdown('<p class="main-title">ℹ️ About This System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Smart Student Risk Prediction & Recommendation System</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Models Used")
        st.markdown("""
        | Model | Purpose |
        |-------|--------|
        | Logistic Regression | Baseline classifier |
        | Random Forest | Main prediction model |
        | KNN | Recommendation system |
        """)
        
        st.markdown("### 📊 Data Split")
        st.markdown("""
        - **Training:** 60%
        - **Validation:** 20%
        - **Testing:** 20%
        """)
    
    with col2:
        st.markdown("### 📋 Features Used")
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | G1, G2 | Period grades |
        | studytime | Weekly study hours |
        | failures | Past class failures |
        | absences | School absences |
        | schoolsup | School support |
        | famsup | Family support |
        | paid | Paid classes |
        | internet | Internet access |
        | higher | Higher education goal |
        | health | Health status |
        | goout | Social activity level |
        """)
    
    st.markdown("---")
    st.info("🎯 **Target:** Risk = 1 if G3 < 10 (At Risk), else 0 (Not At Risk)")


# Page routing
if page == "🎯 Predict Risk":
    show_predict_page()
elif page == "📚 Recommendations":
    show_recommendations_page()
elif page == "📊 Model Performance":
    show_performance_page()
elif page == "ℹ️ About":
    show_about_page()