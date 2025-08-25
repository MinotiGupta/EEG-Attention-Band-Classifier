# EEG Attention State Analyzer - Realtime EEG band monitor

This project is a **Streamlit dashboard** for analyzing EEG signals from `.bdf` files.  
It processes EEG recordings, extracts frequency bands (Delta, Theta, Alpha, Beta, Gamma),  
and maps them to **attentional states** like *Sleepy, Relaxed, Calm, Focused, or Highly Engaged*.  

## Features
- Scan and load `.bdf` EEG files from nested folders
- Preprocess EEG signals (filtering, referencing)
- Compute **frequency band powers** using Welch’s method
- Detect the **dominant brainwave band** and map it to attentional state
- Interactive visualizations with **Plotly**
- User-friendly dashboard with **Streamlit**

## Project Structure
project-root/
│── main.py # Streamlit dashboard entry point
│── data_loader.py # Handles finding & validating .bdf files
│── eeg_processor.py # Preprocessing & frequency band extraction
│── visualization.py # Plotly visualizations for EEG data
│── data/ # Folder containing EEG .bdf files (not in repo)
│── requirements.txt # Dependencies
│── README.md # Project documentation

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows
3. Install requirements:
Python 3.9+, Streamlit >= 1.28.0, MNE, NumPy, SciPy, Plotly
5. Run the app:
streamlit run main.py

## License 
This project is for educational and research purposes only.
EEG data must be collected ethically and in compliance with institutional guidelines.
