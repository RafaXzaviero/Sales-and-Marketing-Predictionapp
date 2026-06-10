import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "best_voting_model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "model_columns.pkl"

FEATURES_TO_SCALE = [
    'age',
    'total_visits',
    'avg_session_time',
    'pages_per_session',
    'email_open_rate',
    'email_click_rate',
    'total_spent',
    'avg_order_value',
    'support_tickets',
    'delivery_delay_days',
    'satisfaction_score',
    'nps_score',
    'marketing_spend_per_user',
    'lifetime_value',
    'last_3_month_purchase_freq',
    'signup_month',
    'signup_day_of_week',
    'last_purchase_month',
    'last_purchase_day_of_week',
    'customer_tenure_days',
]

CATEGORY_PREFIXES = [
    'gender',
    'country',
    'city',
    'acquisition_channel',
    'device_type',
    'subscription_type',
    'coupon_code',
    'payment_method',
]


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, model_columns


def get_category_options(model_columns, prefix):
    options = sorted(
        {col.split(f"{prefix}_", 1)[1] for col in model_columns if col.startswith(f"{prefix}_")}
    )
    if options:
        options.append("Other")
    return options


def build_input_df(model_columns):
    # Numeric user inputs
    data = {
        'age': st.sidebar.slider('Age', 18, 80, 30),
        'total_visits': st.sidebar.slider('Total Visits', 0, 100, 10),
        'avg_session_time': st.sidebar.slider('Average Session Time (minutes)', 0.0, 60.0, 15.0),
        'pages_per_session': st.sidebar.slider('Pages Per Session', 0.0, 20.0, 5.0),
        'email_open_rate': st.sidebar.slider('Email Open Rate', 0.0, 1.0, 0.5),
        'email_click_rate': st.sidebar.slider('Email Click Rate', 0.0, 1.0, 0.1),
        'total_spent': st.sidebar.number_input('Total Spent ($)', 0.0, 10000.0, 500.0),
        'avg_order_value': st.sidebar.number_input('Average Order Value ($)', 0.0, 1000.0, 50.0),
        'support_tickets': st.sidebar.slider('Support Tickets', 0, 20, 1),
        'delivery_delay_days': st.sidebar.slider('Delivery Delay Days', 0, 60, 2),
        'satisfaction_score': st.sidebar.slider('Satisfaction Score (1-5)', 1.0, 5.0, 3.0),
        'nps_score': st.sidebar.slider('NPS Score (0-100)', 0, 100, 50),
        'marketing_spend_per_user': st.sidebar.number_input('Marketing Spend Per User ($)', 0.0, 500.0, 20.0),
        'lifetime_value': st.sidebar.number_input('Lifetime Value ($)', 0.0, 20000.0, 1000.0),
        'last_3_month_purchase_freq': st.sidebar.slider('Last 3 Month Purchase Frequency', 0, 20, 3),
        'signup_month': st.sidebar.slider('Signup Month', 1, 12, 6),
        'signup_day_of_week': st.sidebar.slider('Signup Day of Week (0=Mon, 6=Sun)', 0, 6, 2),
        'last_purchase_month': st.sidebar.slider('Last Purchase Month', 1, 12, 9),
        'last_purchase_day_of_week': st.sidebar.slider('Last Purchase Day of Week (0=Mon, 6=Sun)', 0, 6, 4),
        'customer_tenure_days': st.sidebar.number_input('Customer Tenure (Days)', 0, 365*5, 365),
    }

    # Initialize all model columns to zero
    for col in model_columns:
        if col not in data:
            data[col] = 0

    # Set categorical values from sidebar selections
    for prefix in CATEGORY_PREFIXES:
        options = get_category_options(model_columns, prefix)
        if not options:
            continue
        selection = st.sidebar.selectbox(prefix.replace('_', ' ').title(), options)
        if selection != 'Other':
            col_name = f"{prefix}_{selection}"
            if col_name in data:
                data[col_name] = 1

    features = pd.DataFrame([data], columns=model_columns)
    return features


def main():
    st.set_page_config(page_title='Sales & Marketing Churn Prediction', layout='wide')
    st.title('Sales & Marketing Churn Prediction')
    st.markdown('Input customer details on the sidebar and click Predict to see churn probability.')

    try:
        model, scaler, model_columns = load_artifacts()
    except FileNotFoundError as exc:
        st.error(f"Failed to load deployment artifact: {exc}")
        st.stop()

    st.sidebar.header('Customer Input Features')
    input_df = build_input_df(model_columns)

    st.subheader('User Input Features')
    st.write(input_df)

    if st.button('Predict Churn'):
        input_scaled = input_df.copy()
        input_scaled[FEATURES_TO_SCALE] = scaler.transform(input_scaled[FEATURES_TO_SCALE])

        prediction = model.predict(input_scaled)
        proba = model.predict_proba(input_scaled)

        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The customer is unlikely to churn.')
        else:
            st.warning('The customer is likely to churn.')

        st.subheader('Prediction Probability')
        st.write(f'No Churn: {proba[0][0] * 100:.2f}%')
        st.write(f'Churn: {proba[0][1] * 100:.2f}%')


if __name__ == '__main__':
    main()
