
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMPORTANT:
# Keep True if the saved classifier was trained on X_train_scaled.
# Keep False only if it was trained on X_train_imputed / raw imputed
# features WITHOUT scaling.
MODEL_WAS_TRAINED_ON_SCALED_DATA = True

FEATURES = [
    "customer_id",
    "age",
    "annual_income_inr",
    "credit_score",
    "credit_utilization_ratio",
    "missed_payments_12m",
    "avg_late_payment_days",
    "monthly_transaction_count",
    "monthly_spend_inr",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
    "debt_balance_inr",
]

INTEGER_FEATURES = {
    "customer_id",
    "age",
    "credit_score",
    "missed_payments_12m",
    "monthly_transaction_count",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
}

# Candidate paths allow the app to find common artifact locations.
MODEL_CANDIDATES = [
    MODEL_DIR / "best_model.joblib",
    APP_DIR / "best_model.joblib",
]

IMPUTER_CANDIDATES = [
    MODEL_DIR / "imputer.joblib",
    MODEL_DIR / "knn_imputer.joblib",
]

SCALER_CANDIDATES = [
    MODEL_DIR / "scaler.joblib",
    MODEL_DIR / "standard_scaler.joblib",
]


# ============================================================
# 2. PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 55%, #172554 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1120;
    }

    .hero {
        padding: 28px;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 20px;
        background: linear-gradient(120deg, #172554, #111827);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2.25rem;
    }

    .hero p {
        color: #cbd5e1;
        margin-top: 10px;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 18px;
        border-radius: 15px;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 20px;
        border-radius: 16px;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MODEL AND PREPROCESSING ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    model_path = next(
        (path for path in MODEL_CANDIDATES if path.is_file()),
        None,
    )

    if model_path is None:
        searched = "\n".join(str(p) for p in MODEL_CANDIDATES)
        raise FileNotFoundError(
            "Trained model was not found.\n\n"
            f"Expected one of these paths:\n{searched}"
        )

    model = joblib.load(model_path)

    imputer_path = next(
        (path for path in IMPUTER_CANDIDATES if path.is_file()),
        None,
    )

    scaler_path = next(
        (path for path in SCALER_CANDIDATES if path.is_file()),
        None,
    )

    imputer = joblib.load(imputer_path) if imputer_path else None
    scaler = joblib.load(scaler_path) if scaler_path else None

    return model, imputer, scaler, model_path


try:
    model, imputer, scaler, model_path = load_artifacts()
    artifact_error = None
except Exception as exc:
    model = imputer = scaler = model_path = None
    artifact_error = str(exc)


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛡️ Risk Classifier")
    st.caption("Customer risk prediction dashboard")

    st.divider()

    st.subheader("Model Status")

    if artifact_error:
        st.error("Model loading failed")
        with st.expander("Technical details"):
            st.code(artifact_error)
    else:
        st.success("Model loaded")
        st.caption(f"Artifact: {model_path.name}")

        if hasattr(model, "n_features_in_"):
            st.caption(
                f"Expected features: {model.n_features_in_}"
            )

        st.caption(
            "Preprocessing: "
            + ("Scaled input" if MODEL_WAS_TRAINED_ON_SCALED_DATA
               else "Unscaled input")
        )

    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        "Select page",
        ["Predict Risk", "About Model"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Risk predictions are estimates, not financial decisions.")


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ Risk Alert Classifier</h1>
        <p>
            Supervised machine learning dashboard for customer
            risk assessment using a Random Forest classifier.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. PREDICTION PAGE
# ============================================================

if page == "Predict Risk":

    st.subheader("Customer Risk Assessment")
    st.write(
        "Enter the customer's details below and select "
        "**Predict Risk** to run the saved model."
    )

    with st.form("risk_prediction_form"):

        st.markdown("### Customer Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            customer_id = st.number_input(
                "Customer ID",
                min_value=0,
                value=10001,
                step=1,
            )

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )

            annual_income_inr = st.number_input(
                "Annual Income (INR)",
                min_value=0.0,
                value=600000.0,
                step=10000.0,
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0,
                max_value=1000,
                value=700,
                step=1,
            )

            credit_utilization_ratio = st.number_input(
                "Credit Utilization Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.01,
                format="%.2f",
            )

        with c2:
            missed_payments_12m = st.number_input(
                "Missed Payments (12 Months)",
                min_value=0,
                value=1,
                step=1,
            )

            avg_late_payment_days = st.number_input(
                "Average Late Payment Days",
                min_value=0.0,
                value=5.0,
                step=1.0,
            )

            monthly_transaction_count = st.number_input(
                "Monthly Transaction Count",
                min_value=0,
                value=25,
                step=1,
            )

            monthly_spend_inr = st.number_input(
                "Monthly Spend (INR)",
                min_value=0.0,
                value=25000.0,
                step=1000.0,
            )

            cash_advance_count_6m = st.number_input(
                "Cash Advances (6 Months)",
                min_value=0,
                value=1,
                step=1,
            )

        with c3:
            complaints_last_6m = st.number_input(
                "Complaints (6 Months)",
                min_value=0,
                value=0,
                step=1,
            )

            failed_login_attempts_3m = st.number_input(
                "Failed Login Attempts (3 Months)",
                min_value=0,
                value=0,
                step=1,
            )

            account_tenure_months = st.number_input(
                "Account Tenure (Months)",
                min_value=0,
                value=24,
                step=1,
            )

            debt_balance_inr = st.number_input(
                "Debt Balance (INR)",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )

        st.caption(
            "Gender, region, employment type and last transaction date "
            "are not included in this 14-feature input because the "
            "saved model expects 14 numeric features."
        )

        submitted = st.form_submit_button(
            "🔍 Predict Customer Risk",
            use_container_width=True,
            type="primary",
        )

    if submitted:

        if artifact_error:
            st.error(
                "The model could not be loaded. Fix the model path "
                "or artifact error shown in the sidebar."
            )
            st.stop()

        # IMPORTANT: This order must match model training.
        input_values = {
            "customer_id": customer_id,
            "age": age,
            "annual_income_inr": annual_income_inr,
            "credit_score": credit_score,
            "credit_utilization_ratio": credit_utilization_ratio,
            "missed_payments_12m": missed_payments_12m,
            "avg_late_payment_days": avg_late_payment_days,
            "monthly_transaction_count": monthly_transaction_count,
            "monthly_spend_inr": monthly_spend_inr,
            "cash_advance_count_6m": cash_advance_count_6m,
            "complaints_last_6m": complaints_last_6m,
            "failed_login_attempts_3m": failed_login_attempts_3m,
            "account_tenure_months": account_tenure_months,
            "debt_balance_inr": debt_balance_inr,
        }

        input_df = pd.DataFrame(
            [[input_values[col] for col in FEATURES]],
            columns=FEATURES,
        )

        # Apply the same fitted preprocessing used in training.
        try:
            if len(FEATURES) != getattr(model, "n_features_in_", len(FEATURES)):
                raise ValueError(
                    f"App provides {len(FEATURES)} features, but model "
                    f"expects {model.n_features_in_}."
                )

            processed_input = input_df.copy()

            if imputer is not None:
                processed_input = imputer.transform(processed_input)

            if MODEL_WAS_TRAINED_ON_SCALED_DATA:
                if scaler is None:
                    st.error(
                        "The model is configured for scaled inputs, "
                        "but models/scaler.joblib was not found. "
                        "Save the SAME fitted scaler from your notebook "
                        "to that path before predicting."
                    )
                    st.stop()

                processed_input = scaler.transform(processed_input)

            processed_input = np.asarray(processed_input)

            if processed_input.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"After preprocessing, input has "
                    f"{processed_input.shape[1]} features; model expects "
                    f"{model.n_features_in_}."
                )

            prediction = model.predict(processed_input)[0]

            # Map labels to risk names.
            if str(prediction).strip().lower() in {
                "1", "1.0", "high risk", "high_risk", "high"
            }:
                risk_label = "High Risk"
                is_high_risk = True
            elif str(prediction).strip().lower() in {
                "0", "0.0", "low risk", "low_risk", "low"
            }:
                risk_label = "Low Risk"
                is_high_risk = False
            else:
                risk_label = str(prediction)
                is_high_risk = None

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed_input)[0]
                classes = list(model.classes_)

                high_class_index = next(
                    (
                        i for i, cls in enumerate(classes)
                        if str(cls).strip().lower() in {
                            "1", "1.0", "high risk", "high_risk", "high"
                        }
                    ),
                    None,
                )

                if high_class_index is not None:
                    probability = float(probabilities[high_class_index])

            st.divider()
            st.subheader("Prediction Result")

            result_col, probability_col = st.columns([1, 1])

            with result_col:
                if is_high_risk is True:
                    st.error("🚨 HIGH RISK")
                elif is_high_risk is False:
                    st.success("✅ LOW RISK")
                else:
                    st.info(f"Predicted class: {risk_label}")

                st.caption(
                    "This result is generated by the saved classifier."
                )

            with probability_col:
                if probability is not None:
                    st.metric(
                        "Estimated High-Risk Probability",
                        f"{probability:.1%}",
                    )

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={"suffix": "%"},
                            title={"text": "High-Risk Probability"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#38bdf8"},
                                "steps": [
                                    {"range": [0, 40], "color": "#14532d"},
                                    {"range": [40, 70], "color": "#854d0e"},
                                    {"range": [70, 100], "color": "#7f1d1d"},
                                ],
                            },
                        )
                    )
                    gauge.update_layout(
                        height=260,
                        margin=dict(l=15, r=15, t=45, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#f8fafc",
                    )
                    st.plotly_chart(gauge, use_container_width=True)
                else:
                    st.info(
                        "This model does not provide class probabilities."
                    )

            with st.expander("View submitted feature values"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as exc:
            st.error("Prediction failed.")
            st.write(
                "Check that the input feature order, imputer and scaler "
                "match the training pipeline."
            )
            with st.expander("Technical error details"):
                st.exception(exc)


# ============================================================
# 7. ABOUT MODEL PAGE
# ============================================================

elif page == "About Model":

    st.subheader("About This Classifier")

    st.write(
        "This application loads a previously trained "
        "Random Forest classification model."
    )

    info1, info2, info3 = st.columns(3)

    info1.metric("Expected Input Features", "14")
    info2.metric("Model Type", "Random Forest")
    info3.metric("Task", "Binary Classification")

    st.markdown("### Features used")

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [
            "Customer identifier used by the current saved model",
            "Customer age",
            "Annual income in Indian rupees",
            "Credit score",
            "Credit utilization ratio",
            "Missed payments over 12 months",
            "Average days late on payments",
            "Monthly transaction count",
            "Monthly spending in Indian rupees",
            "Cash advances over 6 months",
            "Complaints over 6 months",
            "Failed login attempts over 3 months",
            "Account tenure in months",
            "Outstanding debt balance in Indian rupees",
        ],
    })

    st.dataframe(feature_table, use_container_width=True)

    st.warning(
        "Important: customer_id is included because the current model "
        "expects 14 features. An identifier is usually not a meaningful "
        "predictor; consider evaluating a model trained without it."
    )

    st.markdown("### Model limitations")
    st.markdown(
        """
        - Predictions depend on the quality of training data.
        - Predicted probabilities are not necessarily calibrated.
        - Performance should be evaluated using recall, precision,
          F1-score, ROC-AUC and a confusion matrix.
        - Do not use this demo alone to make real financial or
          credit decisions.
        """
    )

st.divider()
st.caption(
    "Risk Alert Classifier | Machine Learning Project | "
    "Predictions are informational estimates."
)
