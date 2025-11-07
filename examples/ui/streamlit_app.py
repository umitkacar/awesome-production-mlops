"""
📊 Beautiful Streamlit ML Dashboard
Modern, interactive ML dashboard with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta


# Page configuration
st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def generate_sample_data():
    """Generate sample data for visualization"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    data = pd.DataFrame({
        'date': dates,
        'accuracy': 0.85 + np.random.randn(len(dates)) * 0.05,
        'latency': 100 + np.random.randn(len(dates)) * 20,
        'requests': np.random.randint(1000, 5000, len(dates))
    })

    return data


def main():
    """Main dashboard application"""

    # Title and header
    st.title("🚀 MLOps Dashboard 2024-2025")
    st.markdown("### Monitor your ML models in production with style!")

    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png", use_container_width=True)
        st.markdown("## 🎛️ Controls")

        model_name = st.selectbox(
            "Select Model",
            ["RandomForest v1.0", "XGBoost v2.0", "Neural Network v3.0"]
        )

        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now())
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info(f"**Model:** {model_name}")
        st.info(f"**Version:** 1.0.0")
        st.info(f"**Deployed:** 2024-11-07")

    # Generate data
    data = generate_sample_data()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Avg Accuracy",
            value=f"{data['accuracy'].mean():.2%}",
            delta=f"{np.random.uniform(-0.05, 0.05):.2%}"
        )

    with col2:
        st.metric(
            label="⚡ Avg Latency",
            value=f"{data['latency'].mean():.0f}ms",
            delta=f"{np.random.uniform(-10, 10):.0f}ms"
        )

    with col3:
        st.metric(
            label="📊 Total Requests",
            value=f"{data['requests'].sum():,}",
            delta=f"{np.random.uniform(-1000, 1000):.0f}"
        )

    with col4:
        st.metric(
            label="🎯 Uptime",
            value="99.9%",
            delta="0.1%"
        )

    # Charts
    st.markdown("---")

    # Row 1: Performance over time
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Model Accuracy Over Time")
        fig = px.line(
            data,
            x='date',
            y='accuracy',
            title='Model Accuracy Trend'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚡ Latency Distribution")
        fig = px.histogram(
            data,
            x='latency',
            nbins=50,
            title='Response Time Distribution'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Request volume
    st.markdown("### 📊 Request Volume")
    fig = px.area(
        data,
        x='date',
        y='requests',
        title='Daily Requests'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.markdown("---")
    st.markdown("### 📋 Recent Data")

    st.dataframe(
        data.tail(10).style.background_gradient(cmap='RdYlGn'),
        use_container_width=True
    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ using Streamlit | 🚀 MLOps Ecosystem 2024-2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
