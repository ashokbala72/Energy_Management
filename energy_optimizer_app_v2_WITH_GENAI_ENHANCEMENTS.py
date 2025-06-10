import streamlit as st
import pandas as pd
import numpy as np
import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv
import plotly.express as px

st.set_page_config(page_title="Energy Optimizer", layout="wide")
st.title("⚡ GenAI-Powered Energy Management Dashboard")

# Load OpenAI API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Simulated asset list and tariff
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

def generate_energy_data():
    np.random.seed(42)
    hours = pd.date_range(datetime.datetime.now().replace(minute=0, second=0, microsecond=0), periods=48, freq='H')
    data = pd.DataFrame({
        'timestamp': hours,
        'total_kWh': np.random.randint(120, 180, size=48),
        'temperature_C': np.random.uniform(26, 32, size=48)
    })
    return data

def build_prompt(question, energy_data, assets):
    usage_today = energy_data['total_kWh'].sum()
    max_load_time = energy_data.loc[energy_data['total_kWh'].idxmax(), 'timestamp'].strftime('%H:%M')
    asset_efficiency = {asset: round(np.random.uniform(60, 95), 2) for asset in assets}
    efficiency_str = ', '.join([f'{k}: {v}%' for k, v in asset_efficiency.items()])
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
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

energy_data = generate_energy_data()

overview_tab, main_tab = st.tabs(["📘 Overview", "🏠 Main App"])

with overview_tab:
    st.subheader("📘 Application Overview")
    st.markdown("""
### 🔍 What This App Does

This GenAI-powered Energy Management Dashboard helps industrial and commercial users monitor, analyze, and optimize their energy usage.

Each feature is designed to help reduce operational costs, improve energy efficiency, and support sustainability goals — all through easy-to-use visualizations and AI assistance.

---

### 📚 Tab Descriptions & Benefits

- **📊 Energy Dashboard**  
  View real-time energy consumption across 48 hours.  
  **Benefit:** Helps track usage trends and detect high consumption periods.

- **🧠 GenAI Advisory**  
  Ask AI energy-related questions like how to reduce peak loads.  
  **Benefit:** AI guidance for cost-saving, maintenance, or policy decisions.

- **⚙️ Optimization Engine**  
  Suggests when to run assets based on tariff and load.  
  **Benefit:** Reduce energy costs by running equipment during cheaper periods.

- **💡 Load Forecasting**  
  Shows the next 48-hour forecast of electricity usage.  
  **Benefit:** Enables proactive planning for peak demand.

- **💰 Tariff Planner**  
  Shows time-of-use energy pricing windows.  
  **Benefit:** Understand energy costs throughout the day.

- **🏭 Asset Efficiency**  
  Displays efficiency scores for all key assets.  
  **Benefit:** Identify underperforming equipment.

- **♻️ Carbon Footprint**  
  Shows CO₂ emissions based on usage.  
  **Benefit:** Track carbon output and support sustainability reporting.

- **🔧 Maintenance Impact**  
  Estimates energy loss due to maintenance issues.  
  **Benefit:** Understand how downtime affects efficiency.

- **📄 Executive Summary**  
  GenAI-generated summary of the day’s performance and suggestions.  
  **Benefit:** Fast, AI-driven reporting for decision-makers.

- **📉 Anomaly Alerts**  
  Flags abnormal energy usage patterns.  
  **Benefit:** Catch unexpected spikes in real-time.

- **🧾 Bill Auditor**  
  Simulates billing and flags overcharges.  
  **Benefit:** Detect billing discrepancies early.

- **⚡ Renewable Advisory**  
  Recommends asset scheduling based on solar availability.  
  **Benefit:** Shift load toward renewable-friendly hours.

---

### 🛠️ Steps to Go Production

To take this dashboard live in a real industrial setting:

- ✅ Connect to real-time SCADA or BMS systems via secure APIs  
- ✅ Replace simulated energy data with live metering feeds  
- ✅ Integrate billing and tariff info with ERP systems  
- ✅ Add authentication, access control, and user audit trails  
- ✅ Host on secure, scalable infrastructure (e.g., Azure, AWS, on-prem)

---

### 📦 Data Types

- **Real:**  
  - Tariff structure (sampled from UK TOU tariff logic)  
  - Carbon emission factor (UK grid: 0.233 kg CO₂/kWh)

- **Mock/Simulated:**  
  - Energy usage over 48 hours  
  - Solar availability  
  - Asset constraints & maintenance loss  
  - Asset efficiency and GenAI answers (based on mocked input)

This hybrid setup is ideal for prototyping before live SCADA integration.
    """)

with main_tab:
    tab = st.sidebar.radio("Select a module", [
        "📊 Energy Dashboard", "🧠 GenAI Advisory", "⚙️ Optimization Engine", "💡 Load Forecasting",
        "💰 Tariff Planner", "🏭 Asset Efficiency", "♻️ Carbon Footprint", "🔧 Maintenance Impact",
        "📄 Executive Summary", "📉 Anomaly Alerts", "🧾 Bill Auditor", "⚡ Renewable Advisory"
    ])

    if tab == "📊 Energy Dashboard":
        st.subheader("📊 Real-Time Energy Usage")
        st.dataframe(energy_data)
        fig = px.line(energy_data, x='timestamp', y='total_kWh', title='Energy Usage Over Time')
        fig.update_layout(xaxis_title='Timestamp', yaxis_title='Total kWh')
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
        load_threshold = energy_data['total_kWh'].quantile(0.9)
        peak_hours = energy_data[energy_data['total_kWh'] > load_threshold]['timestamp']
        for asset in assets:
            for period in asset_constraints[asset]:
                start_hour, end_hour = map(lambda t: datetime.datetime.strptime(t, "%H:%M"), period.split("-"))
                candidates = energy_data[
                    (~energy_data['timestamp'].dt.hour.isin(peak_hours.dt.hour)) &
                    (energy_data['timestamp'].dt.hour >= start_hour.hour) &
                    (energy_data['timestamp'].dt.hour < end_hour.hour)
                ]
                if not candidates.empty:
                    best_time = candidates.sort_values(by='total_kWh').iloc[0]['timestamp']
                    schedule.append({"Asset": asset, "Recommended Time": best_time.strftime("%Y-%m-%d %H:%M")})
                    break
        df_sched = pd.DataFrame(schedule)
        st.dataframe(df_sched)

        if not df_sched.empty:
            prompt = f"Based on today's energy data and the tariff schedule, the following run-times are suggested:\n\n{df_sched.to_string(index=False)}\n\nWhat is your recommendation for improving operational energy efficiency further?"
            with st.spinner("Generating GenAI recommendation..."):
                response = get_genai_response(prompt)
                st.markdown("**GenAI Advisory:**")
                st.info(response)

    elif tab == "💡 Load Forecasting":
        st.subheader("💡 48-Hour Load Forecast")
        st.dataframe(energy_data[['timestamp', 'total_kWh']])
        fig = px.line(energy_data, x='timestamp', y='total_kWh', title='48-Hour Load Forecast')
        fig.update_layout(xaxis_title='Timestamp', yaxis_title='Total kWh')
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
        fig = px.line(pd.DataFrame({'timestamp': energy_data['timestamp'], 'Emissions (kg)': emissions.values}), x='timestamp', y='Emissions (kg)', title='Estimated Carbon Emissions')
        fig.update_layout(xaxis_title='Timestamp', yaxis_title='Emissions (kg)')
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
        threshold = energy_data['total_kWh'].mean() + 1.5 * energy_data['total_kWh'].std()
        anomalies = energy_data[energy_data['total_kWh'] > threshold]
        st.dataframe(anomalies)
        if not anomalies.empty:
            prompt = f"Explain causes of these anomalies: {anomalies.to_dict(orient='records')}"
            st.write("**GenAI Analysis:** " + get_genai_response(prompt))

    elif tab == "🧾 Bill Auditor":
        st.subheader("🧾 Bill Reconciliation")
        actual_total = energy_data['total_kWh'].sum()
        billed_total = actual_total * 1.05
        st.metric("Actual Usage", f"{actual_total:.2f} kWh")
        st.metric("Billed", f"{billed_total:.2f} kWh")
        st.metric("Discrepancy", f"{billed_total - actual_total:.2f} kWh")
        prompt = f"The actual was {actual_total:.2f}, but billed was {billed_total:.2f}. Recommend action."
        st.write("**GenAI:** " + get_genai_response(prompt))

    elif tab == "⚡ Renewable Advisory":
        st.subheader("⚡ Renewable Integration")
        simulated_solar = np.clip(np.sin(np.linspace(0, 3.14, 48)) * 100, 0, None)
        energy_data['solar_available_kWh'] = simulated_solar
        fig = px.line(energy_data, x='timestamp', y='solar_available_kWh', title="Simulated Solar Availability")
        fig.update_layout(xaxis_title='Timestamp', yaxis_title='Solar Available kWh')
        st.plotly_chart(fig, use_container_width=True)
        prompt = "Using solar forecast, recommend when to run high-load assets."
        st.write("**GenAI Advisory:** " + get_genai_response(prompt))