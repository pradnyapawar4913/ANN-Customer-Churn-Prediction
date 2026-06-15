import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="FinGuard Bank",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------------------------
# CUSTOM HEADER
# ---------------------------------

st.markdown("""
<div style="
padding:20px;
border-radius:15px;
background:linear-gradient(90deg,#0f172a,#1e3a8a);
color:white;
text-align:center;">
<h1>🏦 FinGuard Bank</h1>
<h3>Customer Retention Analytics Dashboard</h3>
<p>Predict customer churn risk using Artificial Neural Network</p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)

c1.metric("Model","ANN")
c2.metric("Accuracy","86%")
c3.metric("Optimizer","Adam")
c4.metric("Layers","2")


# ---------------------------------
# INPUT SECTION
# ---------------------------------

left,right = st.columns([2,1])

with left:

    st.subheader("📋 Customer Information")

    col1,col2 = st.columns(2)

    with col1:

      geography = st.selectbox(
        'Geography',
        onehot_encoder_geography.categories_[0]
      )

      gender = st.selectbox(
        'Gender',
        label_encoder_gender.classes_
       )

      age = st.slider(
        'Age',
        18,
        92
     )

      tenure = st.slider(
        'Tenure',
        0,
        10
     )

      num_of_products = st.slider(
        'Number of Products',
        1,
        4
     )

    with col2:

       credit_score = st.number_input(
        'Credit Score',
        min_value=300,
        max_value=900,
        value=600
      )

       balance = st.number_input(
        'Balance',
        value=50000.0
     )

       estimated_salary = st.number_input(
        'Estimated Salary',
        value=50000.0
     )

       has_cr_card = st.selectbox(
        'Has Credit Card',
        [0,1]
     )

       is_active_member = st.selectbox(
        'Is Active Member',
        [0,1]
     )

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------

with right:

    st.markdown("## 🎯 Prediction Panel")

    predict_btn = st.button("🔍 Predict Churn Risk")

if predict_btn:

    # ---------------------------------
    # INPUT DATA
    # ---------------------------------

    input_data = pd.DataFrame({
        'CreditScore':[credit_score],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
        'EstimatedSalary':[estimated_salary]
    })

    # ---------------------------------
    # OHE GEOGRAPHY
    # ---------------------------------

    geography_encoded = onehot_encoder_geography.transform(
        [[geography]]
    ).toarray()

    geography_encoded_df = pd.DataFrame(
        geography_encoded,
        columns=onehot_encoder_geography.get_feature_names_out(
            ['Geography']
        )
    )

    # ---------------------------------
    # CONCAT
    # ---------------------------------

    input_data = pd.concat(
        [
            input_data.reset_index(drop=True),
            geography_encoded_df
        ],
        axis=1
    )

    # ---------------------------------
    # SCALING
    # ---------------------------------

    input_scaled = scaler.transform(input_data)

    # ---------------------------------
    # PREDICTION
    # ---------------------------------

    prediction = model.predict(input_scaled)

    prediction_proba = prediction[0][0]

    # ---------------------------------
    # RESULT
    # ---------------------------------

    st.subheader("🎯 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{prediction_proba*100:.2f}%"
    )

    gauge = go.Figure(
      go.Indicator(
        mode="gauge+number",
        value=prediction_proba*100,
        title={"text":"Churn Risk"},
        gauge={
            "axis":{"range":[0,100]},
            "bar":{"color":"red"},
            "steps":[
                {"range":[0,40],"color":"green"},
                {"range":[40,70],"color":"orange"},
                {"range":[70,100],"color":"red"}
            ]
        }
    )
)

    st.plotly_chart(
      gauge,
      use_container_width=True
 )
    st.metric(
    "Churn Probability",
    f"{prediction_proba*100:.2f}%"
)
    # ---------------------------------
    # RISK LEVEL
    # ---------------------------------

    if prediction_proba > 0.7:

        st.error(
            "🔴 HIGH RISK CUSTOMER"
        )

    elif prediction_proba > 0.4:

        st.warning(
            "🟡 MEDIUM RISK CUSTOMER"
        )

    else:

        st.success(
            "🟢 LOW RISK CUSTOMER"
        )

    # ---------------------------------
    # CUSTOMER SNAPSHOT
    # ---------------------------------

    st.subheader("📊 Customer Snapshot")

    m1,m2,m3 = st.columns(3)

    m1.metric(
        "Credit Score",
        credit_score
    )

    m2.metric(
        "Balance",
        f"₹{balance:,.0f}"
    )

    m3.metric(
        "Salary",
        f"₹{estimated_salary:,.0f}"
    )

    # ---------------------------------
    # BAR CHART
    # ---------------------------------

    st.subheader(
        "📈 Customer Financial Profile"
    )

    chart_df = pd.DataFrame({
        "Feature":[
            "Credit Score",
            "Balance",
            "Salary",
            "Age"
        ],
        "Value":[
            credit_score,
            balance,
            estimated_salary,
            age
        ]
    })

    fig = px.bar(
        chart_df,
        x="Feature",
        y="Value",
        title="Customer Profile"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

   
    # ---------------------------------
    # RECOMMENDATION
    # ---------------------------------

    tab1,tab2 = st.tabs([
    "📊 Analytics",
    "💡 Recommendations"
])

    with tab1:
        with tab2:
          st.subheader(
        "💡 Retention Recommendation"
    )
    health_score = (
      credit_score/900*50
      + is_active_member*20
      + has_cr_card*10
      + num_of_products*5
)

    health_score=min(100,health_score)

    st.metric(
       "Financial Health Score",
       f"{health_score:.0f}/100"
)

    if prediction_proba > 0.7:

        st.write("""
        • Contact customer immediately

        • Offer loyalty benefits

        • Assign relationship manager

        • Provide personalized offers
        """)

    elif prediction_proba > 0.4:

        st.write("""
        • Monitor customer activity

        • Increase engagement

        • Recommend suitable products
        """)

    else:

        st.write("""
        • Customer appears stable

        • Maintain current engagement

        • Continue normal support
        """)