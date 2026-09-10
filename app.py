import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# Load Models
# =========================================================

@st.cache_resource
def load_models():

    rul_model = joblib.load("models/rul_pipeline.pkl")
    failure_24h_model = joblib.load("models/failure_24h_pipeline.pkl")
    failure_type_model = joblib.load("models/failure_type_pipeline.pkl")
    repair_cost_model = joblib.load("models/repair_cost_pipeline.pkl")

    return (
        rul_model,
        failure_24h_model,
        failure_type_model,
        repair_cost_model
    )


rul_model, failure_24h_model, failure_type_model, repair_cost_model = load_models()


# =========================================================
# Header
# =========================================================

st.title("🏭 Industrial Predictive Maintenance System")

st.caption(
    "AI-powered monitoring and predictive maintenance for industrial machines."
)

st.divider()


# =========================================================
# Sidebar
# =========================================================

st.sidebar.header("⚙️ Machine Parameters")

machine_id = st.sidebar.number_input(
    "Machine ID",
    min_value=1,
    max_value=20,
    value=7
)

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["CNC", "Pump", "Compressor", "Robotic Arm"]
)

vibration_rms = st.sidebar.number_input(
    "Vibration RMS",
    min_value=0.0,
    max_value=20.0,
    value=4.2,
    step=0.1
)

temperature_motor = st.sidebar.number_input(
    "Motor Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=87.0,
    step=1.0
)

current_phase_avg = st.sidebar.number_input(
    "Current Phase Average",
    min_value=0.0,
    max_value=50.0,
    value=12.5,
    step=0.1
)

pressure_level = st.sidebar.number_input(
    "Pressure Level",
    min_value=0.0,
    max_value=20.0,
    value=6.8,
    step=0.1
)

rpm = st.sidebar.number_input(
    "RPM",
    min_value=0,
    max_value=6000,
    value=3200,
    step=100
)

operating_mode = st.sidebar.selectbox(
    "Operating Mode",
    ["normal", "peak"]
)

hours_since_maintenance = st.sidebar.number_input(
    "Hours Since Maintenance",
    min_value=0.0,
    max_value=1000.0,
    value=140.0,
    step=10.0
)

ambient_temp = st.sidebar.number_input(
    "Ambient Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=27.0,
    step=1.0
)


predict_button = st.sidebar.button(
    "🔍 Predict Machine Health",
    use_container_width=True
)

reset_button = st.sidebar.button(
    "🔄 Reset",
    use_container_width=True
)

if reset_button:
    st.rerun()


# =========================================================
# Prediction
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Input Data
    # -----------------------------------------------------

    input_data = pd.DataFrame([{
        "machine_id": machine_id,
        "machine_type": machine_type,
        "vibration_rms": vibration_rms,
        "temperature_motor": temperature_motor,
        "current_phase_avg": current_phase_avg,
        "pressure_level": pressure_level,
        "rpm": rpm,
        "operating_mode": operating_mode,
        "hours_since_maintenance": hours_since_maintenance,
        "ambient_temp": ambient_temp
    }])


    # -----------------------------------------------------
    # RUL Prediction
    # -----------------------------------------------------

    rul_prediction = float(
        rul_model.predict(input_data)[0]
    )


    # -----------------------------------------------------
    # Failure Within 24h Prediction
    # -----------------------------------------------------

    raw_failure_24h = int(
        failure_24h_model.predict(input_data)[0]
    )


    # =====================================================
    # Failure Type Probabilities
    # =====================================================

    failure_type_probabilities = failure_type_model.predict_proba(
        input_data
    )[0]

    failure_type_classes = failure_type_model.classes_


    # Mapping of encoded classes
    failure_type_mapping = {
        0: "Bearing",
        1: "Electrical",
        2: "Hydraulic",
        3: "Motor Overheat",
        4: "None"
    }


    # Convert probabilities into readable dictionary

    type_probabilities = {}

    for class_value, probability in zip(
        failure_type_classes,
        failure_type_probabilities
    ):

        class_value = int(class_value)

        class_name = failure_type_mapping.get(
            class_value,
            "Unknown"
        )

        type_probabilities[class_name] = float(
            probability
        )


    # =====================================================
    # Final Business Logic
    # =====================================================

    # RUL is used as the final immediate-failure rule.

    if rul_prediction <= 24:

        failure_status = "YES"


        # -------------------------------------------------
        # Choose Failure Type
        # -------------------------------------------------

        real_failure_types = [
            "Bearing",
            "Electrical",
            "Hydraulic",
            "Motor Overheat"
        ]


        # Get probabilities only for real failure types

        real_failure_probabilities = {
            failure_type: type_probabilities.get(
                failure_type,
                0.0
            )

            for failure_type in real_failure_types
        }


        # Select the highest probability real failure

        final_failure_type = max(
            real_failure_probabilities,
            key=real_failure_probabilities.get
        )


        failure_type_confidence = (
            real_failure_probabilities[
                final_failure_type
            ]
        )


        # -------------------------------------------------
        # Repair Cost
        # -------------------------------------------------

        repair_cost_prediction = float(
            repair_cost_model.predict(input_data)[0]
        )

        repair_cost_prediction = max(
            0.0,
            repair_cost_prediction
        )


    else:

        failure_status = "NO"

        final_failure_type = "None"

        failure_type_confidence = 1.0

        repair_cost_prediction = 0.0


    # =====================================================
    # Machine Health
    # =====================================================

    if rul_prediction <= 6:

        health_status = "CRITICAL"

        health_message = (
            "Immediate maintenance is strongly recommended."
        )

        st.error(
            "🚨 CRITICAL CONDITION"
        )

    elif rul_prediction <= 24:

        health_status = "WARNING"

        health_message = (
            "Maintenance should be scheduled soon."
        )

        st.warning(
            "⚠️ WARNING CONDITION"
        )

    else:

        health_status = "HEALTHY"

        health_message = (
            "Machine condition appears stable."
        )

        st.success(
            "✅ HEALTHY"
        )


    st.write(health_message)

    st.divider()


    # =====================================================
    # Prediction Overview
    # =====================================================

    st.subheader("📋 Prediction Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Remaining Useful Life",
            f"{rul_prediction:.1f} h"
        )

    with col2:

        st.metric(
            "Failure Within 24h",
            failure_status
        )

    with col3:

        st.metric(
            "Failure Type",
            final_failure_type
        )

    with col4:

        st.metric(
            "Estimated Repair Cost",
            f"${repair_cost_prediction:,.2f}"
        )


    # =====================================================
    # Failure Type Confidence
    # =====================================================

    if failure_status == "YES":

        st.write("### 🎯 Failure Type Confidence")

        st.progress(
            failure_type_confidence
        )

        st.write(
            f"**{final_failure_type}: "
            f"{failure_type_confidence * 100:.1f}%**"
        )


    st.divider()


    # =====================================================
    # RUL Section
    # =====================================================

    st.subheader("⏳ Remaining Useful Life")

    if rul_prediction <= 6:

        st.error(
            f"Only {rul_prediction:.1f} hours of estimated useful life remain."
        )

    elif rul_prediction <= 24:

        st.warning(
            f"Approximately {rul_prediction:.1f} hours of useful life remain."
        )

    else:

        st.success(
            f"Approximately {rul_prediction:.1f} hours of useful life remain."
        )


    rul_progress = min(
        max(rul_prediction / 100, 0.0),
        1.0
    )

    st.progress(rul_progress)


    st.divider()


    # =====================================================
    # Sensor Monitoring
    # =====================================================

    st.subheader("📊 Sensor Monitoring")

    sensor_col1, sensor_col2, sensor_col3, sensor_col4 = st.columns(4)

    with sensor_col1:

        st.metric(
            "🌡️ Motor Temperature",
            f"{temperature_motor:.1f} °C"
        )

    with sensor_col2:

        st.metric(
            "📳 Vibration RMS",
            f"{vibration_rms:.2f}"
        )

    with sensor_col3:

        st.metric(
            "⚡ Current",
            f"{current_phase_avg:.2f}"
        )

    with sensor_col4:

        st.metric(
            "🔧 Pressure",
            f"{pressure_level:.2f}"
        )


    sensor_col5, sensor_col6, sensor_col7 = st.columns(3)

    with sensor_col5:

        st.metric(
            "🔄 RPM",
            f"{rpm:,}"
        )

    with sensor_col6:

        st.metric(
            "🛠️ Hours Since Maintenance",
            f"{hours_since_maintenance:.1f} h"
        )

    with sensor_col7:

        st.metric(
            "🌡️ Ambient Temperature",
            f"{ambient_temp:.1f} °C"
        )


    st.divider()


    # =====================================================
    # Failure Probability Breakdown
    # =====================================================

    if failure_status == "YES":

        st.subheader("📈 Failure Type Probabilities")

        probability_data = pd.DataFrame({
            "Failure Type": list(type_probabilities.keys()),
            "Probability": [
                probability * 100
                for probability in type_probabilities.values()
            ]
        })

        probability_data = probability_data.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_data.set_index(
                "Failure Type"
            )
        )


        st.divider()


    # =====================================================
    # Machine Information
    # =====================================================

    st.subheader("🏭 Machine Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.write("**Machine ID**")
        st.write(machine_id)

        st.write("**Machine Type**")
        st.write(machine_type)


    with info_col2:

        st.write("**Operating Mode**")
        st.write(operating_mode)

        st.write("**Maintenance Hours**")
        st.write(
            f"{hours_since_maintenance:.1f} h"
        )


    with info_col3:

        st.write("**Health Status**")
        st.write(health_status)

        st.write("**Prediction Time**")
        st.write(
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


    st.divider()


    # =====================================================
    # Maintenance Recommendation
    # =====================================================

    st.subheader("🔧 Maintenance Recommendation")

    if rul_prediction <= 6:

        st.error(
            "🚨 Immediate maintenance required. "
            "The machine has a critically low estimated RUL."
        )

    elif rul_prediction <= 24:

        st.warning(
            "⚠️ Schedule maintenance within the next 24 hours "
            "to reduce the risk of machine failure."
        )

    else:

        st.success(
            "✅ No immediate maintenance required. "
            "Continue monitoring machine sensors."
        )


    st.divider()


    # =====================================================
    # AI Models
    # =====================================================

    with st.expander("🤖 AI Models Used"):

        st.write("### 1️⃣ Remaining Useful Life")

        st.write(
            "**Random Forest Regressor**"
        )

        st.write(
            "Predicts the estimated remaining operating hours "
            "of the machine."
        )


        st.write("### 2️⃣ Failure Within 24h")

        st.write(
            "**XGBoost Classifier**"
        )

        st.write(
            "Predicts whether the machine is likely to experience "
            "a failure within the next 24 hours."
        )


        st.write("### 3️⃣ Failure Type")

        st.write(
            "**XGBoost Multiclass Classifier**"
        )

        st.write(
            "Predicts the most likely failure type using "
            "the probability distribution of the five classes."
        )


        st.write("### 4️⃣ Repair Cost")

        st.write(
            "**Random Forest Regressor**"
        )

        st.write(
            "Estimates the expected maintenance or repair cost."
        )


    # =====================================================
    # Technical Information
    # =====================================================

    with st.expander("ℹ️ Technical Information"):

        st.write(
            "The system uses four independent machine-learning "
            "pipelines trained on industrial machine sensor data."
        )

        st.write(
            "The predicted RUL is used as the final business "
            "rule for determining immediate failure risk."
        )

        st.write(
            "When RUL ≤ 24 hours, the system activates the "
            "failure analysis stage."
        )

        st.write(
            "If the Failure Type model assigns the highest "
            "probability to 'None', the system selects the "
            "highest-probability real failure type instead."
        )

        st.write(
            "Repair cost is only displayed when the machine "
            "is considered at risk of failure."
        )


else:

    # =====================================================
    # Empty State
    # =====================================================

    st.info(
        "👈 Enter the machine parameters from the sidebar "
        "and click **Predict Machine Health** to analyze the machine."
    )