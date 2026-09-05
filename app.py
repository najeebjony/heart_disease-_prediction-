"""
================================================================================
Heart Disease Prediction System - Streamlit Web Dashboard
================================================================================
A production-ready, interactive web application for real-time cardiovascular
risk prediction, machine learning model benchmarking, and clinical EDA.
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioGuard | Heart Disease Prediction AI by Najeeb Ullah",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clinical dashboard UI
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .risk-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        border: 1px solid #c3e6cb;
        text-align: center;
    }
    .risk-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        border: 1px solid #f5c6cb;
        text-align: center;
    }
    .risk-badge-mod {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 700;
        border: 1px solid #ffeeba;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Helper Functions & Model Loader
# -----------------------------------------------------------------------------
# MODEL_PATH = os.path.join("models", "heart_disease_model.pkl")
# DATA_PATH = os.path.join("data", "heart.csv")
BASE_DIR = os.path.dirname(__file__)

# Folder ka naam ("models") hata diya hai kyunke file direct root par hai
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")

@st.cache_resource
def load_model_bundle():
    """Load serialized model bundle or train on the fly if missing."""
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            return bundle
        except Exception as e:
            st.error(f"Error loading saved model: {e}")
    
    # Fallback auto-training if model file not found
    st.warning("Model bundle not found. Training model automatically...")
    from train_model import run_pipeline
    run_pipeline()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    """Load heart disease dataset for EDA and benchmarks."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

# Load model bundle and dataset
bundle = load_model_bundle()
df_data = load_dataset()

champion_model = bundle.get("model")
scaler = bundle.get("scaler")
best_model_name = bundle.get("best_model_name", "Random Forest")
feature_names = bundle.get("feature_names", [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
])
all_models = bundle.get("all_models", {})
metrics_list = bundle.get("metrics", [])
confusion_matrices = bundle.get("confusion_matrices", {})
feature_importances = bundle.get("feature_importances", {})

# Dynamic calculation fallback if artifacts are missing from bundle
if not confusion_matrices and df_data is not None and scaler is not None and all_models:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix
        X = df_data.drop(columns=['target'])
        y = df_data['target'].astype(int)
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_test_scaled = scaler.transform(X_test)
        for name, mdl in all_models.items():
            y_pred = mdl.predict(X_test_scaled)
            confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
    except Exception:
        pass

if not feature_importances and all_models:
    try:
        for name, mdl in all_models.items():
            if hasattr(mdl, "feature_importances_"):
                feature_importances[name] = dict(zip(feature_names, [round(float(v), 4) for v in mdl.feature_importances_]))
            elif hasattr(mdl, "coef_"):
                feature_importances[name] = dict(zip(feature_names, [round(float(abs(v)), 4) for v in mdl.coef_[0]]))
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 3. Sidebar Patient Input Controls
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/heart-with-pulse.png",
    width=70
)
st.sidebar.markdown("## 🫀 CardioGuard AI")
st.sidebar.markdown("👨‍💻 **Developer:** `Najeeb Ullah`")
st.sidebar.markdown("---")
st.sidebar.markdown("## 🩺 Patient Clinical Inputs")
st.sidebar.markdown("Enter the patient's diagnostic parameters below:")

with st.sidebar:
    # Demographics
    st.subheader("1. Demographics")
    age = st.slider("Age (years)", min_value=20, max_value=85, value=52, step=1, help="Patient age in years")
    sex_label = st.radio("Biological Sex", options=["Male", "Female"], horizontal=True)
    sex = 1.0 if sex_label == "Male" else 0.0

    st.markdown("---")
    st.subheader("2. Chest Pain & Cardiac Symptoms")
    cp_options = {
        "Typical Angina (Substernal chest pain)": 1.0,
        "Atypical Angina (Non-classical chest pain)": 2.0,
        "Non-Anginal Pain (Non-cardiac chest discomfort)": 3.0,
        "Asymptomatic (No chest discomfort)": 4.0
    }
    cp_selected = st.selectbox(
        "Chest Pain Type (cp)",
        options=list(cp_options.keys()),
        index=0,
        help="Type of chest pain reported by patient"
    )
    cp = cp_options[cp_selected]

    exang_label = st.radio(
        "Exercise-Induced Angina (exang)",
        options=["No (0)", "Yes (1)"],
        horizontal=True,
        help="Did exercise induce angina/chest pain?"
    )
    exang = 1.0 if "Yes" in exang_label else 0.0

    st.markdown("---")
    st.subheader("3. Vital Signs & Blood Chemistry")
    trestbps = st.slider(
        "Resting Blood Pressure (trestbps) [mm Hg]",
        min_value=80,
        max_value=220,
        value=128,
        step=1,
        help="Resting blood pressure measured upon admission (Normal < 120 mm Hg)"
    )
    chol = st.slider(
        "Serum Cholesterol (chol) [mg/dl]",
        min_value=100,
        max_value=570,
        value=215,
        step=1,
        help="Serum cholesterol in mg/dl (Desirable < 200 mg/dl)"
    )
    fbs_label = st.radio(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        options=["False (<= 120 mg/dl)", "True (> 120 mg/dl)"],
        horizontal=True,
        help="Indicates potential diabetes or hyperglycemia"
    )
    fbs = 1.0 if "True" in fbs_label else 0.0

    st.markdown("---")
    st.subheader("4. ECG & Exercise Stress Metrics")
    restecg_options = {
        "Normal (0)": 0.0,
        "ST-T Wave Abnormality (1)": 1.0,
        "Left Ventricular Hypertrophy (2)": 2.0
    }
    restecg_selected = st.selectbox("Resting ECG Results (restecg)", options=list(restecg_options.keys()), index=0)
    restecg = restecg_options[restecg_selected]

    thalach = st.slider(
        "Maximum Heart Rate Achieved (thalach) [bpm]",
        min_value=60,
        max_value=220,
        value=155,
        step=1,
        help="Highest heart rate achieved during exercise stress test"
    )
    oldpeak = st.slider(
        "ST Depression Induced by Exercise (oldpeak)",
        min_value=0.0,
        max_value=6.5,
        value=0.8,
        step=0.1,
        help="ST depression relative to resting baseline"
    )
    slope_options = {
        "Upsloping (1) - Normal exercise response": 1.0,
        "Flat (2) - Typical ischemic response": 2.0,
        "Downsloping (3) - Severe ischemic response": 3.0
    }
    slope_selected = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=list(slope_options.keys()), index=0)
    slope = slope_options[slope_selected]

    st.markdown("---")
    st.subheader("5. Imaging & Thalassemia")
    ca = st.slider(
        "Number of Major Vessels Colored by Fluoroscopy (ca)",
        min_value=0,
        max_value=3,
        value=0,
        step=1,
        help="Number of major vessels (0-3) visualized with contrast flouroscopy"
    )
    thal_options = {
        "Normal (3)": 3.0,
        "Fixed Defect (6)": 6.0,
        "Reversible Defect (7)": 7.0
    }
    thal_selected = st.selectbox("Thalassemia Status (thal)", options=list(thal_options.keys()), index=0)
    thal = thal_options[thal_selected]

    st.markdown("---")
    predict_btn = st.button("🚀 PREDICT HEART DISEASE RISK", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. Main Page Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🫀 CardioGuard: AI Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Cardiovascular Clinical Decision Support | Developed by <b>Najeeb Ullah</b> (Powered by <b>{best_model_name} ML Pipeline</b>)</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Prediction Engine",
    "📊 Model Analytics & Benchmark",
    "🔬 Exploratory Data Analysis",
    "📖 Clinical Reference Guide"
])

# -----------------------------------------------------------------------------
# TAB 1: Real-Time Prediction Engine
# -----------------------------------------------------------------------------
with tab1:
    # Prepare patient input feature array
    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)

    # Perform prediction
    prediction = champion_model.predict(input_scaled)[0]
    if hasattr(champion_model, "predict_proba"):
        probabilities = champion_model.predict_proba(input_scaled)[0]
        prob_healthy = probabilities[0] * 100
        prob_disease = probabilities[1] * 100
    else:
        prob_disease = 75.0 if prediction == 1 else 25.0
        prob_healthy = 100.0 - prob_disease

    # Prediction Outcome Presentation
    col_pred_left, col_pred_right = st.columns([1.1, 1.3])

    with col_pred_left:
        st.markdown("### 📋 Prediction Diagnostic Result")
        
        if prediction == 1:
            if prob_disease >= 75:
                st.markdown(f'<div class="risk-badge-high">🚨 HIGH RISK: Heart Disease Detected<br><span style="font-size:0.9rem">Disease Probability: {prob_disease:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge-mod">⚠️ MODERATE RISK: Heart Disease Indicated<br><span style="font-size:0.9rem">Disease Probability: {prob_disease:.1f}%</span></div>', unsafe_allow_html=True)
            st.error(f"**Clinical Assessment**: The model classified this profile as **Class 1 (Heart Disease Present)** with a **{prob_disease:.1f}%** statistical probability. Further cardiological evaluation (Echocardiography, Coronary Angiogram) is strongly advised.")
        else:
            if prob_disease <= 25:
                st.markdown(f'<div class="risk-badge-low">✅ LOW RISK: No Heart Disease Detected<br><span style="font-size:0.9rem">Disease Probability: {prob_disease:.1f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge-mod">ℹ️ BORDERLINE LOW RISK: No Overt Disease<br><span style="font-size:0.9rem">Disease Probability: {prob_disease:.1f}%</span></div>', unsafe_allow_html=True)
            st.success(f"**Clinical Assessment**: The model classified this profile as **Class 0 (No Heart Disease Detected)** with a **{prob_healthy:.1f}%** probability of normal cardiac status. Maintain routine preventive health screenings.")

        # Key Patient Metrics summary
        st.markdown("#### Patient Snapshot:")
        snap_col1, snap_col2, snap_col3 = st.columns(3)
        snap_col1.metric("Age / Sex", f"{int(age)}y / {sex_label}")
        snap_col2.metric("BP / Chol", f"{int(trestbps)} / {int(chol)}")
        snap_col3.metric("Max HR", f"{int(thalach)} bpm")

    with col_pred_right:
        st.markdown("### 📈 Disease Risk Probability Gauge")
        
        gauge_color = "#e74c3c" if prob_disease > 60 else ("#f39c12" if prob_disease > 35 else "#2ecc71")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_disease,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Heart Disease Probability ({best_model_name})", 'font': {'size': 18, 'color': '#2c3e50'}},
            number={'suffix': "%", 'font': {'size': 32, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#dddddd",
                'steps': [
                    {'range': [0, 35], 'color': '#d4edda'},
                    {'range': [35, 65], 'color': '#fff3cd'},
                    {'range': [65, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # Risk Factor Breakdown & Recommendations
    col_factors, col_recomms = st.columns(2)

    with col_factors:
        st.markdown("### ⚠️ Clinical Risk Factor Audit")
        risk_flags = []
        normal_flags = []

        if trestbps >= 140:
            risk_flags.append(f"🔴 **High Resting BP ({int(trestbps)} mm Hg)**: Stage 2 Hypertension threshold (≥ 140 mm Hg).")
        elif trestbps >= 130:
            risk_flags.append(f"🟡 **Elevated Resting BP ({int(trestbps)} mm Hg)**: Stage 1 Hypertension threshold (130-139 mm Hg).")
        else:
            normal_flags.append(f"🟢 **Resting BP ({int(trestbps)} mm Hg)**: Within normal limits (< 130 mm Hg).")

        if chol >= 240:
            risk_flags.append(f"🔴 **High Serum Cholesterol ({int(chol)} mg/dl)**: High risk threshold (≥ 240 mg/dl).")
        elif chol >= 200:
            risk_flags.append(f"🟡 **Borderline Cholesterol ({int(chol)} mg/dl)**: Borderline threshold (200-239 mg/dl).")
        else:
            normal_flags.append(f"🟢 **Serum Cholesterol ({int(chol)} mg/dl)**: Desirable range (< 200 mg/dl).")

        if oldpeak >= 1.5:
            risk_flags.append(f"🔴 **Elevated ST Depression ({oldpeak:.1f})**: Suggests significant myocardial ischemia.")
        else:
            normal_flags.append(f"🟢 **ST Depression ({oldpeak:.1f})**: Minimal or no exercise-induced depression.")

        if ca > 0:
            risk_flags.append(f"🔴 **Fluoroscopy Major Vessels ({int(ca)})**: Number of blocked vessels visible ({int(ca)}).")
        else:
            normal_flags.append("🟢 **Fluoroscopy (0 vessels)**: No major vessels showing fluoroscopy defects.")

        if exang == 1.0:
            risk_flags.append("🔴 **Exercise Induced Angina**: Patient experiences exertional chest angina.")

        if fbs == 1.0:
            risk_flags.append("🟡 **Fasting Blood Sugar > 120 mg/dl**: Hyperglycemia / Pre-diabetes risk factor.")

        if risk_flags:
            for rf in risk_flags:
                st.markdown(rf)
        else:
            st.markdown("🎉 *No major elevated risk flags identified based on standard reference ranges.*")

        if normal_flags:
            with st.expander("View Normal Parameters"):
                for nf in normal_flags:
                    st.markdown(nf)

    with col_recomms:
        st.markdown("### 💡 Recommended Clinical Next Steps")
        if prediction == 1:
            st.markdown("""
            1. **Cardiology Consultation**: Schedule a formal evaluation with a certified cardiologist.
            2. **Diagnostic Confirmation**: Perform 12-lead resting/stress ECG, 2D Echocardiogram, or CT Coronary Angiogram.
            3. **Lipid & Glycemic Management**: Statin therapy and dietary adjustments for cholesterol and glycemic control.
            4. **Lifestyle Interventions**: Mediterranean cardiac diet, smoking cessation, and medically supervised aerobic activity.
            """)
        else:
            st.markdown("""
            1. **Preventive Wellness**: Maintain routine annual wellness exams and cardiovascular check-ups.
            2. **Dietary Guidelines**: Low sodium (< 2,300 mg/day), high fiber, lean protein, and heart-healthy unsaturated fats.
            3. **Physical Exercise**: At least 150 minutes of moderate-intensity aerobic physical activity per week.
            4. **Stress & Sleep Management**: Prioritize 7-8 hours of quality sleep and evidence-based stress reduction practices.
            """)

# -----------------------------------------------------------------------------
# TAB 2: Model Performance & Benchmarking
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("### 📊 Multi-Model Performance Benchmarking")
    st.markdown("Comparison of 4 machine learning classifiers evaluated on the stratified test partition (20% holdout):")

    if metrics_list:
        metrics_df_disp = pd.DataFrame(metrics_list)
        # Format table nicely
        st.dataframe(
            metrics_df_disp.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#c3e6cb'),
            use_container_width=True
        )

        # Plotly Comparative Bar Chart
        melted = metrics_df_disp.melt(
            id_vars=['Model'],
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            var_name='Metric',
            value_name='Score'
        )

        fig_comp = px.bar(
            melted,
            x='Metric',
            y='Score',
            color='Model',
            barmode='group',
            title='Model Comparison Across Standard Evaluation Metrics',
            text_auto='.3f',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_comp.update_layout(yaxis_range=[0.7, 1.02], height=420)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # Confusion Matrix & Feature Importance
        col_cm, col_fi = st.columns(2)

        with col_cm:
            st.markdown("### 🔲 Confusion Matrix")
            if confusion_matrices:
                cm_options = list(confusion_matrices.keys())
                selected_cm_model = st.selectbox("Select Model for Confusion Matrix:", options=cm_options, index=0)
                if selected_cm_model in confusion_matrices:
                    cm_vals = np.array(confusion_matrices[selected_cm_model])
                    fig_cm = px.imshow(
                        cm_vals,
                        text_auto=True,
                        labels=dict(x="Predicted Diagnosis", y="Actual Diagnosis", color="Cases"),
                        x=['Healthy (0)', 'Heart Disease (1)'],
                        y=['Healthy (0)', 'Heart Disease (1)'],
                        color_continuous_scale='Blues',
                        title=f'{selected_cm_model} Confusion Matrix'
                    )
                    fig_cm.update_layout(height=380)
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Confusion matrix data unavailable.")

        with col_fi:
            st.markdown("### 🌲 Feature Importance Ranking")
            if feature_importances:
                avail_fi_models = list(feature_importances.keys())
                selected_fi_model = st.selectbox("Select Model for Feature Importances:", options=avail_fi_models, index=0)
                if selected_fi_model in feature_importances:
                    fi_data = feature_importances[selected_fi_model]
                    fi_df = pd.DataFrame(list(fi_data.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)

                    fig_fi = px.bar(
                        fi_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'{selected_fi_model} Feature Importance',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        template='plotly_white'
                    )
                    fig_fi.update_layout(height=380)
                    st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance data unavailable.")

# -----------------------------------------------------------------------------
# TAB 3: Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🔬 Interactive Dataset Exploration & Analytics")
    if df_data is not None:
        st.markdown(f"**Cleveland Heart Disease Dataset Preview** ({df_data.shape[0]} patient records, {df_data.shape[1]} variables):")
        
        # Filter controls
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            target_filter = st.multiselect(
                "Filter by Disease Status:",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda x: "Heart Disease (1)" if x == 1 else "Healthy (0)"
            )
        with filter_col2:
            sex_filter = st.multiselect(
                "Filter by Sex:",
                options=[0.0, 1.0],
                default=[0.0, 1.0],
                format_func=lambda x: "Male (1)" if x == 1.0 else "Female (0)"
            )

        filtered_df = df_data[(df_data['target'].isin(target_filter)) & (df_data['sex'].isin(sex_filter))]
        st.dataframe(filtered_df.head(100), use_container_width=True)

        st.markdown("---")

        # Distribution Plots
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            eda_num_var = st.selectbox("Select Numerical Feature to Plot:", options=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
            fig_hist = px.histogram(
                df_data,
                x=eda_num_var,
                color=df_data['target'].map({0: 'Healthy (0)', 1: 'Heart Disease (1)'}),
                marginal='box',
                barmode='overlay',
                title=f'Distribution of {eda_num_var.upper()} by Heart Disease Status',
                color_discrete_map={'Healthy (0)': '#2ecc71', 'Heart Disease (1)': '#e74c3c'},
                template='plotly_white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_eda2:
            st.markdown("#### Correlation Heatmap")
            corr = df_data.corr().round(2)
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                title="Full Feature Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: Medical Glossary & Disclaimer
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("### 📖 Clinical Parameter Glossary & Reference Standards")
    
    st.markdown("""
    | Parameter | Clinical Description | Typical Normal Range | Significance |
    |---|---|---|---|
    | **age** | Patient age in completed years | 20 - 85 yrs | Cardiovascular risk naturally scales with age. |
    | **sex** | Biological sex (1 = Male, 0 = Female) | 0 or 1 | Men historically exhibit higher earlier incidence. |
    | **cp** | Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic) | 1 - 4 | Asymptomatic / atypical angina can mask ischemic disease. |
    | **trestbps** | Resting Blood Pressure on admission | < 120 mm Hg | Hypertension directly damages arterial endothelium. |
    | **chol** | Serum cholesterol in mg/dl | < 200 mg/dl | Elevated LDL promotes atherosclerosis. |
    | **fbs** | Fasting Blood Sugar > 120 mg/dl | <= 120 mg/dl (0) | Marker for insulin resistance / diabetes. |
    | **restecg** | Resting electrocardiographic results | Normal (0) | Identifies baseline electrical conduction abnormalities. |
    | **thalach** | Maximum heart rate achieved under stress | 130 - 180 bpm | Chronotropic incompetence indicates vascular impairment. |
    | **exang** | Exercise induced angina | 0 (No) | Exertional chest pain indicates coronary hypoperfusion. |
    | **oldpeak** | ST depression induced by exercise vs rest | < 1.0 mm | Significant ST depression indicates reversible ischemia. |
    | **slope** | Slope of peak exercise ST segment | Upsloping (1) | Downsloping/Flat ST indicates coronary disease. |
    | **ca** | Major coronary vessels colored by fluoroscopy | 0 | Higher vessel count correlates with multivessel disease. |
    | **thal** | Thallium stress scintigraphy | Normal (3) | Fixed/reversible perfusion defects indicate ischemia/infarction. |
    """)

    st.markdown("---")
    st.warning("""
    **Medical Disclaimer**: This web application and machine learning model are designed strictly for educational, research, and clinical decision support purposes. This system is **not** a substitute for professional medical diagnosis, advice, or treatment. Always seek the advice of a qualified physician or healthcare provider with any questions regarding a medical condition.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666666; font-size: 0.95rem;'>🫀 <b>CardioGuard AI</b> © 2026 | Developed by <b>Najeeb Ullah</b> | Built with Python, Scikit-Learn, Streamlit & Plotly</p>", unsafe_allow_html=True)
