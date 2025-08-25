import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import mne
from typing import Dict


class Visualizer:
    """Handles visualization of EEG data and analysis results"""

    def __init__(self):
        # Color scheme for frequency bands
        self.band_colors = {
            'Delta': '#1f77b4',
            'Theta': '#ff7f0e',
            'Alpha': '#2ca02c',
            'Beta': '#d62728',
            'Gamma': '#9467bd'
        }

        # Attention state colors
        self.attention_colors = {
            'Sleepy': '#3498db',
            'Relaxed': '#2ecc71',
            'Calm': '#f39c12',
            'Focused': '#e74c3c',
            'Highly engaged': '#9b59b6'
        }

    def plot_band_powers(self, band_powers: Dict[str, float], dominant_band: str) -> go.Figure:
        """
        Create a bar chart of frequency band powers with enhanced styling

        Args:
            band_powers: Dictionary of band powers
            dominant_band: Name of dominant band

        Returns:
            Plotly figure
        """
        bands = list(band_powers.keys())
        powers = list(band_powers.values())

        # Create colors list, highlighting dominant band
        colors = []
        for band in bands:
            if band == dominant_band:
                colors.append('#ff6b6b')  # Highlight color
            else:
                colors.append(self.band_colors.get(band, '#gray'))

        fig = go.Figure(data=[
            go.Bar(
                x=bands,
                y=powers,
                marker_color=colors,
                text=[f'{p:.2f}' for p in powers],
                textposition='auto',
                hovertemplate='<b>%{x} Band</b><br>Power: %{y:.2f} μV²<extra></extra>'
            )
        ])

        fig.update_layout(
            title={
                'text': 'EEG Frequency Band Powers',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Frequency Bands',
            yaxis_title='Power (μV²)',
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        # Add annotation for dominant band
        max_power = max(powers)
        dominant_idx = bands.index(dominant_band)

        fig.add_annotation(
            x=dominant_idx,
            y=max_power * 1.1,
            text=f"🎯 Dominant: {dominant_band}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ff6b6b",
            arrowwidth=2,
            bgcolor="white",
            bordercolor="#ff6b6b",
            borderwidth=2,
            font=dict(size=12, color="#ff6b6b")
        )

        return fig

    def plot_raw_eeg(self, raw: mne.io.Raw, duration: float = 10.0, n_channels: int = 8) -> go.Figure:
        """
        Plot raw EEG data for a specified duration with improved styling

        Args:
            raw: MNE Raw object
            duration: Duration to plot in seconds
            n_channels: Number of channels to display

        Returns:
            Plotly figure
        """
        # Get data for specified duration
        start_time = 0
        end_time = min(duration, raw.times[-1])

        # Select subset of channels for visualization
        n_channels = min(n_channels, len(raw.ch_names))
        selected_channels = raw.ch_names[:n_channels]

        # Get data
        data, times = raw[selected_channels,
                      raw.time_as_index(start_time)[0]:raw.time_as_index(end_time)[0]]

        # Create figure
        fig = go.Figure()

        # Color palette for channels
        colors = px.colors.qualitative.Set3[:n_channels]

        # Add traces for each channel
        for i, (ch_data, ch_name) in enumerate(zip(data, selected_channels)):
            # Offset channels vertically for better visualization
            offset = i * 100  # Increased offset for better separation

            fig.add_trace(go.Scatter(
                x=times,
                y=ch_data + offset,
                mode='lines',
                name=ch_name,
                line=dict(width=1.5, color=colors[i % len(colors)]),
                hovertemplate=f'<b>{ch_name}</b><br>Time: %{{x:.2f}}s<br>Amplitude: %{{customdata:.2f}} μV<extra></extra>',
                customdata=ch_data
            ))

        fig.update_layout(
            title={
                'text': f'Live EEG Data Stream ({duration:.1f}s window)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Time (s)',
            yaxis_title='Channels (μV)',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        # Remove y-axis ticks since channels are offset
        fig.update_yaxis(showticklabels=False)

        return fig

    def plot_power_spectrum(self, raw: mne.io.Raw) -> go.Figure:
        """
        Plot power spectral density

        Args:
            raw: MNE Raw object

        Returns:
            Plotly figure
        """
        # Compute PSD
        spectrum = raw.compute_psd(method='welch', fmax=50, verbose=False)
        freqs = spectrum.freqs
        psd_data = spectrum.get_data()

        # Average across channels
        psd_mean = np.mean(psd_data, axis=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=freqs,
            y=10 * np.log10(psd_mean),  # Convert to dB
            mode='lines',
            name='Average PSD',
            line=dict(width=2)
        ))

        # Add vertical lines for frequency bands
        for band_name, (low_freq, high_freq) in {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }.items():
            fig.add_vline(x=low_freq, line_dash="dash",
                          annotation_text=band_name,
                          line_color=self.band_colors.get(band_name, 'gray'))

        fig.update_layout(
            title='Power Spectral Density',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power (dB)',
            height=400
        )

        return fig

    def plot_attention_history(self, history_df: pd.DataFrame) -> go.Figure:
        """
        Plot attention state history over time

        Args:
            history_df: DataFrame with attention state history

        Returns:
            Plotly figure
        """

        # Map attention states to numeric values for plotting
        state_mapping = {
            'Sleepy': 1,
            'Relaxed': 2,
            'Calm': 3,
            'Focused': 4,
            'Highly engaged': 5
        }

        history_df['attention_numeric'] = history_df['attention_state'].map(state_mapping)

        fig = go.Figure()

        # Add line plot
        fig.add_trace(go.Scatter(
            x=history_df['time'],
            y=history_df['attention_numeric'],
            mode='lines+markers',
            name='Attention Level',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            hovertemplate='<b>%{customdata}</b><br>Time: %{x}<br>Level: %{y}<extra></extra>',
            customdata=history_df['attention_state']
        ))

        # Add colored background regions
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
        states = ['Sleepy', 'Relaxed', 'Calm', 'Focused', 'Highly engaged']

        for i, (state, color) in enumerate(zip(states, colors)):
            fig.add_hrect(
                y0=i + 0.5, y1=i + 1.5,
                fillcolor=color, opacity=0.1,
                layer="below", line_width=0,
            )

        fig.update_layout(
            title={
                'text': 'Attention State Timeline',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Time',
            yaxis_title='Attention Level',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 6)),
                ticktext=states,
                range=[0.5, 5.5]
            ),
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        return fig

    def create_realtime_gauge(self, attention_state: str) -> go.Figure:
        """
        Create a gauge chart for real-time attention monitoring

        Args:
            attention_state: Current attention state

        Returns:
            Plotly figure
        """
        # Map attention states to gauge values
        state_values = {
            'Sleepy': 1,
            'Relaxed': 2,
            'Calm': 3,
            'Focused': 4,
            'Highly engaged': 5
        }

        value = state_values.get(attention_state, 3)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Attention Level: {attention_state}"},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': self.attention_colors.get(attention_state, 'gray')},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "#3498db"},
                    {'range': [2, 3], 'color': "#2ecc71"},
                    {'range': [3, 4], 'color': "#f39c12"},
                    {'range': [4, 5], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5
                }
            }
        ))

        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))

        return fig
