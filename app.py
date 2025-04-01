# Welcome to your Streamlit Shell!
import streamlit as st
import os
import pandas as pd
import numpy as np
import warnings

from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# need absolute path to ml_depend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPEND_DIR = os.path.join(BASE_DIR, "ml_depend")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Title Page", "Graphs", "Machine Learning"])

# Title Page
if page == "Title Page":
    st.title("")

# Graphs Page
elif page == "Graphs":
    st.title("")

# Machine Learning Page
######################## DO NOT EDIT THIS SECTION ##############################
elif page == "Machine Learning":
    st.markdown("<h1 style='text-align: center;'>GTIN Sales Forecasting Using META Prophet</h1>", unsafe_allow_html=True)

    # Info Section
    st.markdown("""
    **Model**: [Facebook Prophet](https://facebook.github.io/prophet/) — a time series forecasting model based on additive regression.

    **Data**: Daily packaged beverage sales data per GTIN from convenience stores in Idaho during 2023 sourced from PDI C_Store data. The model forecasts weekly total quantity sold for each GTIN based on historical sales.
    """)

    @st.cache_data
    def load_data():
        file_path = os.path.join(DEPEND_DIR, "idaho_target_2023.parquet")
        df = pd.read_parquet(file_path)
        df['ds'] = pd.to_datetime(df['DAY_OF_PREDICTION'])
        return df

    def tune_prophet_for_gtin(gtin, group, param_grid):
        group = group.sort_values(by="ds")
        if len(group) < 4:
            return {"GTIN": gtin, "predicted": 0, "actual": group['TOTAL_QUANTITY'].iloc[-1] if len(group) > 0 else None, "best_params": None, "mae": None}

        train = group.iloc[:-1]
        test = group.iloc[-1:]
        train_ts = train[['ds', 'TOTAL_QUANTITY']].rename(columns={'TOTAL_QUANTITY': 'y'})
        actual = test['TOTAL_QUANTITY'].iloc[0]

        best_mae = float('inf')
        best_pred = None
        best_params = None

        for cps in param_grid['changepoint_prior_scale']:
            for sps in param_grid['seasonality_prior_scale']:
                try:
                    model = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                    model.fit(train_ts)
                    future = model.make_future_dataframe(periods=1, freq='W')
                    forecast = model.predict(future)
                    pred = forecast['yhat'].iloc[-1]
                    mae = abs(pred - actual)
                    if mae < best_mae:
                        best_mae = mae
                        best_pred = pred
                        best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps}
                except:
                    continue

        return {"GTIN": gtin, "predicted": best_pred, "actual": actual, "best_params": best_params, "mae": best_mae}

    def train_and_tune_models(df):
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05],
            'seasonality_prior_scale': [0.1, 1]
        }

        results = []
        grouped = list(df.groupby("GTIN"))
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(tune_prophet_for_gtin, gtin, group, param_grid) for gtin, group in grouped]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    st.error(f"Error tuning model for a GTIN: {e}")

        results_df = pd.DataFrame(results).sort_values(by="GTIN")
        results_df.to_csv(os.path.join(DEPEND_DIR, "predictions_prophet.csv"), index=False)

        valid_results = results_df[results_df['mae'].notnull()]
        if valid_results.empty:
            st.warning("No GTINs with sufficient data for evaluation.")
        return results_df

    def train_global_model(df):
        agg_df = df.groupby('ds')['TOTAL_QUANTITY'].sum().reset_index()
        agg_df = agg_df.rename(columns={'TOTAL_QUANTITY': 'y'})
        model = Prophet()
        model.fit(agg_df)
        return model, agg_df

    # Sidebar training options
    st.sidebar.header("Model Training Options")
    model_filename = os.path.join(DEPEND_DIR, "prophet_trained_model.json")
    use_existing = False

    if os.path.exists(model_filename):
        choice = st.sidebar.radio("Trained model exists. Use it or retrain?", ["Use existing model", "Retrain model"])
        use_existing = (choice == "Use existing model")
    else:
        st.sidebar.warning("No trained global model found. You must train a new one.")
        use_existing = False

    df = load_data()

    if use_existing:
        with open(model_filename, 'r') as fin:
            global_model = model_from_json(fin.read())
        st.success("Loaded existing trained global model.")
    else:
        with st.spinner("Training models. This may take some time..."):
            predictions_df = train_and_tune_models(df)
            st.success("GTIN-level models trained and predictions saved.")

            st.write("---")
            st.write("Training global aggregated model...")
            global_model, agg_df = train_global_model(df)
            with open(model_filename, 'w') as fout:
                fout.write(model_to_json(global_model))
            st.success("Global model trained and saved.")

    if 'global_model' in locals():
        future = global_model.make_future_dataframe(periods=1, freq='W')
        forecast = global_model.predict(future)
        forecast_val = forecast['yhat'].iloc[-1]
        st.subheader("Global Sales Forecast")
        st.write(f"Next week's forecasted total sales: **{round(forecast_val)} units**")

    # GTIN Selection
    st.write("---")
    st.subheader("Select GTIN to View Prediction")
    try:
        pred_file = os.path.join(DEPEND_DIR, "predictions_prophet.csv")
        predictions_df = pd.read_csv(pred_file, dtype={"GTIN": str})
        gtin_list = predictions_df["GTIN"].sort_values().unique().tolist()
        selected_gtin = st.selectbox("Select GTIN", gtin_list)
        result = predictions_df[predictions_df["GTIN"] == selected_gtin]
        pred = result.iloc[0]["predicted"]
        if pd.isna(pred) or pred == 0:
            st.warning("Insufficient data for prediction.")
        else:
            st.success(f"GTIN #{selected_gtin}: Predicted sales next week: {round(pred)} units")
    except FileNotFoundError:
        st.warning("Prediction file not found. Please train the model first.")
