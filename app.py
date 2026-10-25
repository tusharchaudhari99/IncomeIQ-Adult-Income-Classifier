import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import html

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IncomeIQ | AI Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "tuned_xgboost_income_model.pkl"


# =========================================================
# 2. LOAD THE TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.info(
        "Check that the model file exists and that the installed "
        "scikit-learn and XGBoost versions match the training environment."
    )
    st.stop()


# =========================================================
# 3. ANIMATED DASHBOARD CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #19284b 0%, transparent 42%),
            radial-gradient(ellipse at 95% 15%, #30204b 0%, transparent 38%),
            #0b1020;
        color: #f4f7ff;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121a2e, #0d1323);
        border-right: 1px solid #283650;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes floatIn {
        from {
            opacity: 0;
            transform: translateY(14px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 10px #42d9ff12; }
        50% { box-shadow: 0 0 24px #42d9ff30; }
    }

    @keyframes growBar {
        from { width: 0; }
    }

    .hero {
        background: linear-gradient(
            115deg,
            #172748,
            #34265c,
            #123d55,
            #172748
        );
        background-size: 300% 300%;
        animation: gradientShift 12s ease infinite;
        border: 1px solid #4a638a;
        border-radius: 24px;
        padding: 30px 32px;
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: clamp(28px, 3vw, 42px);
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -1.3px;
        color: white;
        margin: 8px 0;
    }

    .hero-subtitle {
        color: #d0ddf7;
        font-size: 15px;
        line-height: 1.7;
        max-width: 760px;
    }

    .eyebrow {
        color: #79e7ff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #f4f7ff;
        margin: 14px 0 5px 0;
    }

    .section-subtitle {
        color: #aab7d0;
        font-size: 13px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(145deg, #18243b, #121b2e);
        border: 1px solid #2a3a58;
        border-radius: 18px;
        padding: 21px;
        min-height: 130px;
        animation: floatIn 0.7s ease both, glow 5s ease-in-out infinite;
        margin-bottom: 10px;
    }

    .metric-label {
        color: #aab7d0;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: white;
        font-size: clamp(21px, 2.3vw, 30px);
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .metric-caption {
        color: #92a4c5;
        font-size: 11px;
        margin-top: 8px;
    }

    .panel {
        background: linear-gradient(145deg, #151f34, #101829);
        border: 1px solid #293954;
        border-radius: 20px;
        padding: 22px;
        margin: 8px 0 18px 0;
        animation: floatIn 0.65s ease both;
    }

    .result-banner {
        border-radius: 16px;
        padding: 20px;
        margin: 12px 0 18px 0;
        background: linear-gradient(110deg, #123c38, #163047);
        border: 1px solid #278b83;
        animation: floatIn 0.6s ease both;
    }

    .result-banner.low {
        background: linear-gradient(110deg, #18344b, #232747);
        border-color: #4777a9;
    }

    .result-label {
        color: #b7d8e9;
        font-size: 12px;
        font-weight: 600;
    }

    .result-value {
        color: white;
        font-size: clamp(24px, 3vw, 34px);
        font-weight: 800;
        margin-top: 8px;
    }

    .badge {
        display: inline-block;
        border: 1px solid #536b91;
        background: #263653;
        color: #d9e8ff;
        border-radius: 20px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 700;
    }

    .prob-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        color: #dce7fb;
        font-size: 13px;
        margin: 17px 0 8px 0;
    }

    .prob-track {
        width: 100%;
        height: 12px;
        background: #26334b;
        border-radius: 20px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #42d9ff, #9b8cff);
        animation: growBar 1.2s ease both;
    }

    .prob-fill.high {
        background: linear-gradient(90deg, #36dfbd, #42d9ff);
    }

    .footer {
        text-align: center;
        color: #7f91b0;
        font-size: 11px;
        padding: 20px 0 5px 0;
        border-top: 1px solid #283650;
        margin-top: 25px;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(110deg, #2878e8, #7458e8, #2878e8);
        background-size: 200% 200%;
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 12px 20px;
        font-weight: 750;
        min-height: 46px;
        transition: transform 0.2s ease, filter 0.2s ease;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        filter: brightness(1.15);
        transform: translateY(-2px);
        color: white;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
    }

    [data-testid="stMetric"] {
        background: #151f34;
        border: 1px solid #293954;
        padding: 14px;
        border-radius: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #aab7d0;
    }

    [data-testid="stMetricValue"] {
        color: #f4f7ff;
    }

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label {
        color: #d7e3f8 !important;
        font-weight: 600 !important;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">MACHINE LEARNING • CLASSIFICATION</div>
        <div class="hero-title">IncomeIQ Dashboard</div>
        <div class="hero-subtitle">
            Explore an income category prediction powered by your trained
            XGBoost model. Enter profile details, generate a prediction,
            and review the model's estimated class probabilities.
        </div>
        <br>
        <span class="badge">⚡ XGBoost</span>
        &nbsp;
        <span class="badge">📊 Binary Classification</span>
        &nbsp;
        <span class="badge">✨ Interactive Dashboard</span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📊 IncomeIQ")
    st.caption("Adult Census Income Classifier")
    st.markdown("---")

    st.markdown("### 🧠 Model Information")
    st.write("**Algorithm:** XGBoost")
    st.write("**Task:** Binary classification")
    st.write("**Output:** Income category")

    st.markdown("---")
    st.markdown("### 🎯 Prediction Classes")
    st.markdown("**Class 0:** $50K or less")
    st.markdown("**Class 1:** More than $50K")

    st.markdown("---")
    st.caption(
        "Class labels assume 0 = <=50K and 1 = >50K. "
        "Verify the target encoding in your training notebook."
    )


# =========================================================
# 6. INPUT FORM
# =========================================================

st.markdown(
    '<div class="section-title">👤 Profile & Employment Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Enter the information used by your trained model.'
    '</div>',
    unsafe_allow_html=True
)

with st.form("income_prediction_form"):

    personal_tab, work_tab, finance_tab = st.tabs(
        [
            "👤 Personal & Education",
            "💼 Employment",
            "💰 Financial Details"
        ]
    )

    with personal_tab:
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input(
                "Age",
                min_value=17,
                max_value=100,
                value=33
            )

            education = st.selectbox(
                "Education",
                [
                    "Bachelors", "Some-college", "11th", "HS-grad",
                    "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                    "7th-8th", "12th", "Masters", "1st-4th",
                    "10th", "Doctorate", "5th-6th", "Preschool"
                ],
                index=6
            )

            education_num = st.number_input(
                "Education Number",
                min_value=1,
                max_value=16,
                value=10
            )

        with c2:
            sex = st.selectbox("Sex", ["Male", "Female"])

            race = st.selectbox(
                "Race",
                [
                    "White", "Asian-Pac-Islander",
                    "Amer-Indian-Eskimo", "Other", "Black"
                ]
            )

            relationship = st.selectbox(
                "Relationship",
                [
                    "Wife", "Own-child", "Husband", "Not-in-family",
                    "Other-relative", "Unmarried"
                ]
            )

            marital_status = st.selectbox(
                "Marital Status",
                [
                    "Married-civ-spouse", "Divorced",
                    "Never-married", "Separated", "Widowed",
                    "Married-spouse-absent", "Married-AF-spouse"
                ]
            )

    with work_tab:
        c1, c2 = st.columns(2)

        with c1:
            workclass = st.selectbox(
                "Workclass",
                [
                    "Private", "Self-emp-not-inc", "Self-emp-inc",
                    "Federal-gov", "Local-gov", "State-gov",
                    "Without-pay", "Never-worked"
                ],
                index=2
            )

            occupation = st.selectbox(
                "Occupation",
                [
                    "Tech-support", "Craft-repair", "Other-service",
                    "Sales", "Exec-managerial", "Prof-specialty",
                    "Handlers-cleaners", "Machine-op-inspct",
                    "Adm-clerical", "Farming-fishing",
                    "Transport-moving", "Priv-house-serv",
                    "Protective-serv", "Armed-Forces"
                ]
            )

        with c2:
            hours_per_week = st.number_input(
                "Hours per Week",
                min_value=1,
                max_value=100,
                value=40
            )

            fnlwgt = st.number_input(
                "Final Weight (fnlwgt)",
                min_value=1,
                value=20000
            )

            native_country = st.selectbox(
                "Native Country",
                [
                    "United-States", "Mexico", "Philippines",
                    "Germany", "Canada", "India", "England",
                    "Cuba", "Jamaica", "China", "Italy",
                    "South", "Puerto-Rico", "El-Salvador",
                    "Japan", "Vietnam"
                ]
            )

    with finance_tab:
        c1, c2 = st.columns(2)

        with c1:
            capital_gain = st.number_input(
                "Capital Gain",
                min_value=0,
                value=0
            )

        with c2:
            capital_loss = st.number_input(
                "Capital Loss",
                min_value=0,
                value=0
            )

        st.info(
            "Enter the capital gain and capital loss values "
            "for the profile you want to evaluate."
        )

    submitted = st.form_submit_button(
        "⚡ Generate Income Prediction",
        use_container_width=True
    )


# =========================================================
# 7. PREDICTION AND RESULTS
# =========================================================

if submitted:

    input_data = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education.num": education_num,
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital.gain": capital_gain,
        "capital.loss": capital_loss,
        "hours.per.week": hours_per_week,
        "native.country": native_country
    }])

    try:
        prediction = model.predict(input_data)[0]

        if not hasattr(model, "predict_proba"):
            st.error(
                "This model does not support predict_proba()."
            )
            st.stop()

        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        def class_is(label, expected):
            try:
                return float(label) == expected
            except (TypeError, ValueError):
                return str(label).strip() == str(expected)

        class0_index = next(
            (
                i for i, label in enumerate(classes)
                if class_is(label, 0)
            ),
            None
        )

        class1_index = next(
            (
                i for i, label in enumerate(classes)
                if class_is(label, 1)
            ),
            None
        )

        if class0_index is None or class1_index is None:
            st.error(
                f"Unexpected model classes: {classes}. "
                "Check target encoding in your training notebook."
            )
            st.stop()

        p0 = float(probabilities[class0_index])
        p1 = float(probabilities[class1_index])

        if class_is(prediction, 0):
            result_title = "$50K or less"
            result_detail = "Predicted class: 0"
            result_style = "low"
            result_icon = "📘"

        elif class_is(prediction, 1):
            result_title = "More than $50K"
            result_detail = "Predicted class: 1"
            result_style = ""
            result_icon = "📈"

        else:
            st.error(f"Unexpected prediction value: {prediction}")
            st.stop()

        confidence = max(p0, p1) * 100

        st.markdown(
            '<div class="section-title">📊 Prediction Dashboard</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Results generated by your saved machine learning model.'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-banner {result_style}">
                <div class="result-label">
                    {result_icon} PREDICTED INCOME CATEGORY
                </div>
                <div class="result-value">
                    {html.escape(result_title)}
                </div>
                <div class="result-label">{result_detail}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dashboard metric cards
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">CLASS 0 PROBABILITY</div>
                    <div class="metric-value">{p0:.2%}</div>
                    <div class="metric-caption">
                        Income $50K or less*
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">CLASS 1 PROBABILITY</div>
                    <div class="metric-value">{p1:.2%}</div>
                    <div class="metric-caption">
                        Income above $50K*
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">MODEL CONFIDENCE</div>
                    <div class="metric-value">{confidence:.2f}%</div>
                    <div class="metric-caption">
                        Highest predicted class probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Animated probability bars
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">
                    📈 Class Probability Analysis
                </div>
                <div class="section-subtitle">
                    Estimated probability for each model class.
                </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="prob-row">
                <span>Income $50K or less (Class 0)</span>
                <strong>{p0:.2%}</strong>
            </div>
            <div class="prob-track">
                <div class="prob-fill"
                     style="width:{p0 * 100:.2f}%"></div>
            </div>

            <div class="prob-row">
                <span>Income above $50K (Class 1)</span>
                <strong>{p1:.2%}</strong>
            </div>
            <div class="prob-track">
                <div class="prob-fill high"
                     style="width:{p1 * 100:.2f}%"></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Submitted profile
        with st.expander("🔎 Review submitted profile"):
            st.dataframe(
                input_data.T.rename(
                    columns={0: "Submitted value"}
                ),
                use_container_width=True
            )

        st.caption(
            "*Class meanings assume 0 = <=50K and 1 = >50K. "
            "Confirm this mapping in your training notebook. "
            "Probabilities are model estimates, not guarantees."
        )

    except Exception as error:
        st.error("Prediction could not be completed.")
        st.code(str(error))
        st.info(
            "Check the input columns, categorical values, "
            "preprocessing pipeline, and package versions."
        )

else:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">🚀 Ready to predict?</div>
            <div class="section-subtitle">
                Complete the form and select
                <b>Generate Income Prediction</b> to view the
                result, probabilities, and submitted profile.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">MODEL</div>
                <div class="metric-value">XGBoost</div>
                <div class="metric-caption">Trained classifier</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">INPUT FEATURES</div>
                <div class="metric-value">14</div>
                <div class="metric-caption">
                    Profile and employment data
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">OUTPUT CLASSES</div>
                <div class="metric-value">2</div>
                <div class="metric-caption">
                    Two income categories
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 8. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        IncomeIQ • Adult Census Income Classification • XGBoost
        <br><br>
        Educational demonstration only. Predictions may reflect
        historical dataset biases and should not be used to make
        consequential decisions about individuals.
    </div>
    """,
    unsafe_allow_html=True
)