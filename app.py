import streamlit as st
import pandas as pd
import pickle


# =========================================================
# LOAD MODEL + SCALER
# =========================================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================
   COMPLETE PAGE BACKGROUND
   ========================= */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.35), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(168,85,247,0.35), transparent 30%),
        radial-gradient(circle at 20% 90%, rgba(20,184,166,0.30), transparent 30%),
        radial-gradient(circle at 90% 90%, rgba(236,72,153,0.25), transparent 30%),
        linear-gradient(135deg, #e0f2fe, #f5f3ff, #ecfdf5);

    min-height: 100vh;
}


/* =========================
   MAIN CONTAINER
   ========================= */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================
   HEADER
   ========================= */

.header-box {
    background:
        linear-gradient(
            135deg,
            #2563eb 0%,
            #7c3aed 45%,
            #db2777 100%
        );

    padding: 35px 30px;
    border-radius: 25px;

    text-align: center;
    color: white;

    box-shadow:
        0 15px 40px rgba(79,70,229,0.30);

    margin-bottom: 28px;
}

.header-box h1 {
    font-size: 42px;
    margin: 0;
    font-weight: 800;
}

.header-box p {
    font-size: 17px;
    margin-top: 10px;
    opacity: 0.92;
}


/* =========================
   INPUT CARD
   ========================= */

.input-card {
    background: rgba(255,255,255,0.78);

    border: 1px solid rgba(255,255,255,0.9);

    border-radius: 22px;

    padding: 25px;

    box-shadow:
        0 12px 35px rgba(15,23,42,0.08);

    backdrop-filter: blur(12px);

    margin-bottom: 25px;
}


/* =========================
   SECTION TITLE
   ========================= */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #1e293b;

    margin-bottom: 20px;
}


/* =========================
   STREAMLIT INPUT LABEL
   ========================= */

.stNumberInput label {
    font-weight: 700 !important;
    color: #334155 !important;
    font-size: 14px !important;
}


/* =========================
   INPUT BOX
   ========================= */

.stNumberInput input {
    background: rgba(255,255,255,0.95) !important;

    border: 2px solid #dbeafe !important;

    border-radius: 12px !important;

    height: 45px !important;

    font-size: 15px !important;
}

.stNumberInput input:focus {
    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 2px rgba(99,102,241,0.15) !important;
}


/* =========================
   PREDICT BUTTON
   ========================= */

.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {

    width: 320px;

    height: 55px;

    border-radius: 15px;

    border: none;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed,
            #db2777
        );

    color: white;

    font-size: 18px;

    font-weight: 800;

    box-shadow:
        0 10px 25px rgba(99,102,241,0.30);

    transition: all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 15px 30px rgba(99,102,241,0.40);
}


/* =========================
   RESULT CARD
   ========================= */

.result-card {

    margin-top: 30px;

    padding: 35px;

    border-radius: 25px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.95),
            rgba(239,246,255,0.95)
        );

    box-shadow:
        0 15px 40px rgba(15,23,42,0.10);
}


.result-title {

    font-size: 25px;

    font-weight: 800;

    color: #1e293b;

}


.probability {

    font-size: 58px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed,
            #db2777
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin: 8px 0;
}


.result-description {

    color: #64748b;

    font-size: 15px;
}


/* =========================
   INFO CARDS
   ========================= */

.info-box {

    background: rgba(255,255,255,0.72);

    border-radius: 18px;

    padding: 18px;

    text-align: center;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.07);

    margin-top: 20px;
}

.info-number {

    font-size: 22px;

    font-weight: 800;

    color: #4f46e5;
}

.info-label {

    font-size: 13px;

    color: #64748b;
}


/* =========================
   DISCLAIMER
   ========================= */

.disclaimer {

    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #fffbeb
        );

    border-left: 5px solid #f97316;

    border-radius: 14px;

    padding: 18px;

    margin-top: 30px;

    color: #7c2d12;

    font-size: 13px;

}


/* =========================
   MOBILE
   ========================= */

@media (max-width: 700px) {

    .header-box h1 {
        font-size: 30px;
    }

    .header-box {
        padding: 25px 15px;
    }

    .stButton > button {
        width: 100%;
    }

    .probability {
        font-size: 45px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header-box">

    🩺 Diabetes Predictor

    
        Machine Learning Based Diabetes Risk Prediction System
    

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT CARD
# =========================================================

st.markdown("""
<div class="input-card">

<div class="section-title">
📋 Patient Information
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TWO COLUMN INPUT LAYOUT
# =========================================================

col1, col2 = st.columns(2, gap="large")


# LEFT COLUMN
with col1:

    Pregnancies = st.number_input(
        "🤰 Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    Glucose = st.number_input(
        "🩸 Glucose Level",
        min_value=0,
        max_value=300,
        value=120
    )

    BloodPressure = st.number_input(
        "💓 Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    SkinThickness = st.number_input(
        "📏 Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )


# RIGHT COLUMN
with col2:

    Insulin = st.number_input(
        "💉 Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    BMI = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    DiabetesPedigreeFunction = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    Age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

if st.button("🔍  Predict Diabetes"):

    # Create DataFrame
    input_data = pd.DataFrame(
        [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = model.predict(input_scaled)


    # Probability
    probability = model.predict_proba(input_scaled)

    diabetes_probability = probability[0][1] * 100


    # =====================================================
    # RESULT
    # =====================================================

    if prediction[0] == 1:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
                ⚠️ Higher Likelihood of Diabetes
            </div>

            <div class="probability">
                {diabetes_probability:.2f}%
            </div>

            <div class="result-description">
                Estimated diabetes probability according to the ML model
            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
                ✅ Lower Likelihood of Diabetes
            </div>

            <div class="probability">
                {diabetes_probability:.2f}%
            </div>

            <div class="result-description">
                Estimated diabetes probability according to the ML model
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # PROGRESS BAR
    # =====================================================

    st.write("")

    st.progress(
        int(diabetes_probability)
    )


    # =====================================================
    # EXTRA INFORMATION
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="info-box">

            <div class="info-number">
                {diabetes_probability:.2f}%
            </div>

            <div class="info-label">
                Diabetes Probability
            </div>

        </div>
        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""
        <div class="info-box">

            <div class="info-number">
                {100 - diabetes_probability:.2f}%
            </div>

            <div class="info-label">
                No Diabetes Probability
            </div>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">

<b>⚠️ Important Disclaimer</b>

<br><br>

This application is developed for educational and
demonstration purposes only. The prediction is generated
by a machine learning model and should not be considered
a medical diagnosis.

Please consult a qualified healthcare professional
for medical advice.

</div>
""", unsafe_allow_html=True)