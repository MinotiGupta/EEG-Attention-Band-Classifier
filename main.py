import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import time
import threading
from eeg_processor import EEGProcessor
from data_loader import DataLoader
from visualization import Visualizer
from realtime_processor import RealtimeEEGProcessor


def main():
    st.set_page_config(
        page_title="How Attentive Are You? - Real-time EEG Monitor",
        page_icon="🧠",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .attention-state {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .realtime-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: #ff4444;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="main-header">🧠 How Attentive Are You?</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time EEG Attention Monitoring Dashboard</p>', unsafe_allow_html=True)

    # Initialize components
    data_loader = DataLoader()
    eeg_processor = EEGProcessor()
    visualizer = Visualizer()
    realtime_processor = RealtimeEEGProcessor()

    # Set default data directory
    default_data_dir = "C:/Users/minot/Downloads/dataverse_files"

    # Sidebar for controls
    st.sidebar.header("🎛️ Real-time Controls")

    # Data directory (pre-filled with your path)
    data_dir = st.sidebar.text_input(
        "Data Directory Path",
        value=default_data_dir,
        help="Path to directory containing .bdf files"
    )

    # Real-time settings
    st.sidebar.subheader("⚡ Real-time Settings")
    chunk_duration = st.sidebar.slider("Update Interval (seconds)", 1, 10, 3)
    window_size = st.sidebar.slider("Analysis Window (seconds)", 5, 30, 10)

    # Load available files
    if st.sidebar.button("🔍 Scan for EEG files"):
        with st.spinner("Scanning for .bdf files..."):
            bdf_files = data_loader.find_bdf_files(data_dir)
            if bdf_files:
                st.session_state.bdf_files = bdf_files
                st.sidebar.success(f"Found {len(bdf_files)} .bdf files")
            else:
                st.sidebar.error("No .bdf files found in the specified directory")
                st.info(f"📁 Looking for .bdf files in: `{data_dir}`")
                st.info("Please make sure the path is correct and contains .bdf files")

    # File selection and real-time processing
    if 'bdf_files' in st.session_state and st.session_state.bdf_files:
        selected_file = st.sidebar.selectbox(
            "Select EEG file for real-time simulation",
            options=st.session_state.bdf_files,
            format_func=lambda x: str(Path(x).name)
        )

        # Processing parameters
        st.sidebar.subheader("⚙️ Processing Parameters")
        low_freq = st.sidebar.slider("High-pass filter (Hz)", 0.1, 5.0, 1.0, 0.1)
        high_freq = st.sidebar.slider("Low-pass filter (Hz)", 30.0, 100.0, 50.0, 1.0)

        # Real-time control buttons
        col1, col2 = st.sidebar.columns(2)

        with col1:
            start_button = st.button("🚀 Start Real-time", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop", type="secondary")

        # Initialize session state for real-time processing
        if 'realtime_active' not in st.session_state:
            st.session_state.realtime_active = False
        if 'current_attention_state' not in st.session_state:
            st.session_state.current_attention_state = "Initializing..."
        if 'band_powers_history' not in st.session_state:
            st.session_state.band_powers_history = []

        # Handle start/stop buttons
        if start_button:
            st.session_state.realtime_active = True
            st.session_state.selected_file = selected_file
            st.session_state.chunk_duration = chunk_duration
            st.session_state.window_size = window_size
            st.session_state.low_freq = low_freq
            st.session_state.high_freq = high_freq
            st.rerun()

        if stop_button:
            st.session_state.realtime_active = False
            st.rerun()

        # Real-time processing
        if st.session_state.realtime_active:
            run_realtime_dashboard(realtime_processor, visualizer)
        else:
            show_welcome_screen()

    else:
        show_initial_screen(data_dir)


def show_initial_screen(data_dir):
    """Show initial screen when no files are loaded"""

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white;">
            <h2>🎯 Ready to Test Your Attention?</h2>
            <p style="font-size: 1.1rem; margin: 20px 0;">
                Connect your EEG data and discover your real-time attention patterns!
            </p>
            <p style="font-size: 0.9rem; opacity: 0.8;">
                Click "Scan for EEG files" in the sidebar to get started
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Show expected data structure
    st.subheader("📋 Expected Data Structure")
    st.info(f"Looking for .bdf files in: `{data_dir}`")

    st.code(f"""
{data_dir}/
├── subject_01/
│   ├── session_01.bdf
│   ├── session_02.bdf
│   └── ...
├── subject_02/
│   ├── session_01.bdf
│   └── ...
└── individual_files.bdf
    """)

    st.markdown("""
    ### 🔧 Supported Formats
    - **.bdf files** (BioSemi Data Format)
    - **Nested folder structures** supported
    - **Multiple subjects/sessions** automatically detected
    """)


def show_welcome_screen():
    """Show welcome screen when files are loaded but real-time is not active"""

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); border-radius: 20px; color: white;">
            <h2>🚀 Ready for Real-time Analysis!</h2>
            <p style="font-size: 1.1rem; margin: 20px 0;">
                Your EEG files are loaded and ready to go.
            </p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Click "Start Real-time" to begin monitoring your attention state!
            </p>
        </div>
        """, unsafe_allow_html=True)


def run_realtime_dashboard(realtime_processor, visualizer):
    """Run the real-time EEG dashboard"""

    # Real-time indicator
    st.markdown("""
    <div class="realtime-indicator">
        🔴 LIVE
    </div>
    """, unsafe_allow_html=True)

    # Create placeholders for real-time updates
    attention_placeholder = st.empty()
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    raw_data_placeholder = st.empty()
    history_placeholder = st.empty()

    # Initialize real-time processor if not already done
    if 'realtime_initialized' not in st.session_state:
        with st.spinner("🔄 Initializing real-time EEG processing..."):
            try:
                realtime_processor.initialize(
                    st.session_state.selected_file,
                    st.session_state.chunk_duration,
                    st.session_state.window_size,
                    st.session_state.low_freq,
                    st.session_state.high_freq
                )
                st.session_state.realtime_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                st.session_state.realtime_active = False
                st.rerun()

    # Real-time processing loop
    if st.session_state.realtime_initialized:
        # Get next chunk of data
        try:
            chunk_data = realtime_processor.get_next_chunk()

            if chunk_data is not None:
                band_powers, dominant_band, attention_state, raw_chunk = chunk_data

                # Update session state
                st.session_state.current_attention_state = attention_state
                st.session_state.band_powers_history.append({
                    'timestamp': time.time(),
                    'attention_state': attention_state,
                    'dominant_band': dominant_band,
                    **band_powers
                })

                # Keep only last 50 data points for history
                if len(st.session_state.band_powers_history) > 50:
                    st.session_state.band_powers_history = st.session_state.band_powers_history[-50:]

                # Update displays
                update_attention_display(attention_placeholder, attention_state, dominant_band)
                update_metrics_display(metrics_placeholder, band_powers)
                update_band_chart(chart_placeholder, visualizer, band_powers, dominant_band)
                update_raw_data_display(raw_data_placeholder, visualizer, raw_chunk)
                update_history_display(history_placeholder, visualizer)

            else:
                # End of data reached
                st.session_state.realtime_active = False
                st.success("✅ Reached end of EEG data. Click 'Start Real-time' to replay.")
                st.rerun()

        except Exception as e:
            st.error(f"Real-time processing error: {str(e)}")
            st.session_state.realtime_active = False
            st.rerun()

        # Auto-refresh every chunk_duration seconds
        time.sleep(st.session_state.chunk_duration)
        st.rerun()


def update_attention_display(placeholder, attention_state, dominant_band):
    """Update the main attention state display"""

    # Color mapping for attention states
    state_colors = {
        "Sleepy": "#3498db",
        "Relaxed": "#2ecc71",
        "Calm": "#f39c12",
        "Focused": "#e74c3c",
        "Highly engaged": "#9b59b6"
    }

    # Emoji mapping
    state_emojis = {
        "Sleepy": "😴",
        "Relaxed": "😌",
        "Calm": "😊",
        "Focused": "🎯",
        "Highly engaged": "🔥"
    }

    color = state_colors.get(attention_state, "#95a5a6")
    emoji = state_emojis.get(attention_state, "🧠")

    with placeholder.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}44);
            border: 3px solid {color};
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            margin: 20px 0;
        ">
            <h1 style="color: {color}; margin: 0; font-size: 3rem;">
                {emoji} {attention_state}
            </h1>
            <p style="color: #666; font-size: 1.2rem; margin: 10px 0;">
                Dominant Band: <strong>{dominant_band}</strong>
            </p>
            <p style="color: #888; font-size: 0.9rem;">
                Real-time EEG Analysis • Updated every {st.session_state.chunk_duration}s
            </p>
        </div>
        """, unsafe_allow_html=True)


def update_metrics_display(placeholder, band_powers):
    """Update the metrics display"""

    with placeholder.container():
        st.subheader("📊 Current Band Powers")

        cols = st.columns(5)

        for i, (band, power) in enumerate(band_powers.items()):
            with cols[i]:
                st.metric(
                    label=f"{band} Band",
                    value=f"{power:.2f} μV²",
                    delta=None
                )


def update_band_chart(placeholder, visualizer, band_powers, dominant_band):
    """Update the band power chart"""

    with placeholder.container():
        fig = visualizer.plot_band_powers(band_powers, dominant_band)
        fig.update_layout(title="🎵 Real-time Frequency Band Powers")
        st.plotly_chart(fig, use_container_width=True)


def update_raw_data_display(placeholder, visualizer, raw_chunk):
    """Update the raw EEG data display"""

    with placeholder.container():
        st.subheader("📈 Live EEG Signal")
        fig = visualizer.plot_raw_eeg(raw_chunk, duration=st.session_state.window_size)
        fig.update_layout(title="Real-time EEG Data Stream")
        st.plotly_chart(fig, use_container_width=True)


def update_history_display(placeholder, visualizer):
    """Update the attention state history"""

    if len(st.session_state.band_powers_history) > 5:
        with placeholder.container():
            st.subheader("📈 Attention State History")

            # Create history dataframe
            history_df = pd.DataFrame(st.session_state.band_powers_history)
            history_df['time'] = pd.to_datetime(history_df['timestamp'], unit='s')

            # Plot attention state over time
            fig = visualizer.plot_attention_history(history_df)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
