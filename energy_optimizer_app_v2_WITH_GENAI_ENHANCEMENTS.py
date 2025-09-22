import streamlit as st
import pandas as pd
import numpy as np
import datetime
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import plotly.express as px

# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(page_title="Energy Optimizer", layout="wide")
st.title("⚡ GenAI-Powered Energy Management Dashboard")

# -----------------------------
# Load Azure OpenAI Credentials
# -----------------------------
load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-raj"

# Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -----------------------------
# Static Config
# -----------------------------
assets = ["Pump 1", "Pump 2", "Motor A", "Motor B", "Cooling Tower"]
tariff = {
    "00:00-06:00": 2.5,
    "06:00-12:00": 5.0,
    "12:00-18:00": 7.0,
    "18:00-22:00": 10.0,
    "22:00-00:00": 3.5
}
asset_constraints = {
    "Pump 1": ["06:00-12:00", "12:00-18:00"],
    "Pump 2": ["00:00-06:00", "22:00-00:00"],
    "Motor A": ["06:00-12:00"],
    "Motor B": ["12:00-18:00", "18:00-22:00"],
    "Cooling Tower": ["00:00-06:00", "06:00-12:00"]
}

# -----------------------------
# Helper Functions
# -----------------------------
def generate_energy_data():
    np.random.seed(42)
    hours = pd.date_range(
        datetime.datetime.now().replace(minute=0, second=0, microsecond=0),
        periods=48, freq="H"
    )
    data = pd.DataFrame({
        "timestamp": hours,
        "total_kWh": np.random.randint(120, 180, size=48),
        "temperature_C": np.random.uniform(26, 32, size=48),
    })
    return data

def build_prompt(question, energy_data, assets):
    usage_today = energy_data["total_kWh"].sum()
    max_load_time = energy_data.loc[energy_data["total_kWh"].idxmax(), "timestamp"].strftime("%H:%M")
    asset_efficiency = {asset: round(np.random.uniform(60, 95), 2) for asset in assets}
    efficiency_str = ", ".join([f"{k}: {v}%" for k, v in asset_efficiency.items()])
    prompt = (
        f"You are an expert in industrial energy optimization. "
        f"Based on the following operational data, answer the user's question clearly and helpfully.\n\n"
        f"Data:\n"
        f"- Total energy used today: {usage_today:.2f} kWh\n"
        f"- Peak load time: {max_load_time}\n"
        f"- Asset efficiencies: {efficiency_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt

def get_genai_response(prompt):
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Azure OpenAI error: {e}"

# -----------------------------
# Data Simulation
# -----------------------------
energy_data = generate_energy_data()

# -----------------------------
# Streamlit Tabs
# -----------------------------
overview_tab, main_tab = st.tabs(["📘 Overview", "🏠 Main App"])

# -----------------------------
# Overview Tab
# -----------------------------
with overview_tab:
    st.subheader("📘 Application Overview")
    st.markdown("""
This GenAI-powered Energy Management Dashboard helps monitor, analyze, and optimize energy usage for industrial and commercial assets.

**Key Features:**
- 📊 Energy Dashboard (real-time usage)
- 🧠 GenAI Advisory (AI energy guidance)
- ⚙️ Optimization Engine (asset scheduling)
- 💡 Load Forecasting (48h forecast)
- 💰 Tariff Planner (TOU rates)
- 🏭 Asset Efficiency (performance tracking)
- ♻️ Carbon Footprint (CO₂ emissions)
- 🔧 Maintenance Impact (downtime costs)
- 📄 Executive Summary (AI daily report)
- 📉 Anomaly Alerts (abnormal usage detection)
- 🧾 Bill Auditor (billing reconciliation)
- ⚡ Renewable Advisory (solar integration)
    """)

# -----------------------------
# Main Tab with Modules
# -----------------------------
with main_tab:
    tab = st.sidebar.radio(
        "Select a module",
        [
            "📊 Energy Dashboard", "🧠 GenAI Advisory", "⚙️ Optimization Engine", "💡 Load Forecasting",
            "💰 Tariff Planner", "🏭 Asset Efficiency", "♻️ Carbon Footprint", "🔧 Maintenance Impact",
            "📄 Executive Summary", "📉 Anomaly Alerts", "🧾 Bill Auditor", "⚡ Renewable Advisory",
        ],
    )

    if tab == "📊 Energy Dashboard":
        st.subheader("📊 Real-Time Energy Usage")
        st.dataframe(energy_data)
        fig = px.line(energy_data, x="timestamp", y="total_kWh", title="Energy Usage Over Time")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Today's Estimated Cost (£)", f"{energy_data['total_kWh'].sum() * 6:.2f}")

    elif tab == "🧠 GenAI Advisory":
        st.subheader("🧠 Ask GenAI for Energy Guidance")
        question = st.text_input("Ask something like: How to reduce today's peak use?")
        if question:
            with st.spinner("Thinking..."):
                prompt = build_prompt(question, energy_data, assets)
                st.markdown("**GenAI Response:** " + get_genai_response(prompt))

    elif tab == "⚙️ Optimization Engine":
        st.subheader("⚙️ Optimized Asset Scheduling")
        schedule = []
        load_threshold = energy_data["total_kWh"].quantile(0.9)
        peak_hours = energy_data[energy_data["total_kWh"] > load_threshold]["timestamp"]
        for asset in assets:
            for period in asset_constraints[asset]:
                start_hour, end_hour = map(lambda t: datetime.datetime.strptime(t, "%H:%M"), period.split("-"))
                candidates = energy_data[
                    (~energy_data["timestamp"].dt.hour.isin(peak_hours.dt.hour)) &
                    (energy_data["timestamp"].dt.hour >= start_hour.hour) &
                    (energy_data["timestamp"].dt.hour < end_hour.hour)
                ]
                if not candidates.empty:
                    best_time = candidates.sort_values(by="total_kWh").iloc[0]["timestamp"]
                    schedule.append({"Asset": asset, "Recommended Time": best_time.strftime("%Y-%m-%d %H:%M")})
                    break
        df_sched = pd.DataFrame(schedule)
        st.dataframe(df_sched)

        if not df_sched.empty:
            prompt = f"Suggested run-times:\n\n{df_sched.to_string(index=False)}\n\nWhat improvements do you recommend?"
            with st.spinner("Generating GenAI recommendation..."):
                st.info(get_genai_response(prompt))

    elif tab == "💡 Load Forecasting":
        st.subheader("💡 48-Hour Load Forecast")
        fig = px.line(energy_data, x="timestamp", y="total_kWh", title="48-Hour Load Forecast")
        st.plotly_chart(fig, use_container_width=True)

    elif tab == "💰 Tariff Planner":
        st.subheader("💰 Tariff Rates")
        for period, rate in tariff.items():
            st.write(f"{period} → £{rate}/kWh")

    elif tab == "🏭 Asset Efficiency":
        st.subheader("🏭 Asset Energy Efficiency")
        eff_scores = {asset: round(np.random.uniform(60, 95), 2) for asset in assets}
        df = pd.DataFrame(eff_scores.items(), columns=["Asset", "Efficiency (%)"])
        st.dataframe(df)
        st.bar_chart(df.set_index("Asset"))

    elif tab == "♻️ Carbon Footprint":
        st.subheader("♻️ Estimated Carbon Emissions")
        emissions = energy_data["total_kWh"] * 0.233
        fig = px.line(
            pd.DataFrame({"timestamp": energy_data["timestamp"], "Emissions (kg)": emissions.values}),
            x="timestamp", y="Emissions (kg)", title="Estimated Carbon Emissions"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Total Emissions (kg)", f"{emissions.sum():.2f}")

    elif tab == "🔧 Maintenance Impact":
        st.subheader("🔧 Maintenance Impact")
        losses = {asset: round(np.random.uniform(5, 15), 2) for asset in assets}
        df = pd.DataFrame(list(losses.items()), columns=["Asset", "Estimated Energy Loss (kWh)"])
        st.dataframe(df)

    elif tab == "📄 Executive Summary":
        st.subheader("📄 AI Executive Summary")
        summary_prompt = build_prompt("Summarize today's energy use, inefficiencies, and next steps.", energy_data, assets)
        with st.spinner("Generating summary..."):
            st.write(get_genai_response(summary_prompt))

    elif tab == "📉 Anomaly Alerts":
        st.subheader("📉 Energy Anomaly Detection")
        threshold = energy_data["total_kWh"].mean() + 1.5 * energy_data["total_kWh"].std()
        anomalies = energy_data[energy_data["total_kWh"] > threshold]
        st.dataframe(anomalies)
        if not anomalies.empty:
            prompt = f"Explain causes of these anomalies: {anomalies.to_dict(orient='records')}"
            st.info(get_genai_response(prompt))

    elif tab == "🧾 Bill Auditor":
        st.subheader("🧾 Bill Reconciliation")
        actual_total = energy_data["total_kWh"].sum()
        billed_total = actual_total * 1.05
        st.metric("Actual Usage", f"{actual_total:.2f} kWh")
        st.metric("Billed", f"{billed_total:.2f} kWh")
        st.metric("Discrepancy", f"{billed_total - actual_total:.2f} kWh")
        prompt = f"The actual was {actual_total:.2f}, but billed was {billed_total:.2f}. Recommend action."
        st.info(get_genai_response(prompt))

    elif tab == "⚡ Renewable Advisory":
        st.subheader("⚡ Renewable Integration")
        simulated_solar = np.clip(np.sin(np.linspace(0, 3.14, 48)) * 100, 0, None)
        energy_data["solar_available_kWh"] = simulated_solar
        fig = px.line(energy_data, x="timestamp", y="solar_available_kWh", title="Simulated Solar Availability")
        st.plotly_chart(fig, use_container_width=True)
        prompt = "Using solar forecast, recommend when to run high-load assets."
        st.info(get_genai_response(prompt))
