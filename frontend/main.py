import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import altair as alt

from pages import add_patient, view_patients, update_patient_view, upload_video_view, video_results, analysis


# Helper functions to interact with the API


# Sample data function
def get_ankle_positions():
    timestamps = pd.date_range(start="2024-01-01", periods=100, freq='T')
    left_ankle = np.cumsum(np.random.randn(100, 2), axis=0)
    right_ankle = np.cumsum(np.random.randn(100, 2), axis=0)
    data = [{"timestamp": str(timestamps[i]), "left_ankle": left_ankle[i], "right_ankle": right_ankle[i]} for i in
            range(100)]
    return data


def get_predicted_features():
    return {
        "Speed": 0.8,
        "Cadence": 0.6,
        "Knee Flexion": 0.4
    }


st.set_page_config(layout="wide")

with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Add Patient", "View Patients", "Upload Video", "View Video Results", "Analysis"],
        icons=["house", "plus", "list", "upload", "search", 'clipboard-data'],
        menu_icon="cast",
        default_index=0,
    )


if selected == 'Home':
    st.title("Patient Management System")

# Add Patient page
elif selected == "Add Patient":
    add_patient()

# View Patients page
elif selected == "View Patients":
    st.title("View Patients")
    view_patients()

# Update Patient page
elif selected == "Update Patient":
    update_patient_view()

elif selected == "View Video Results":
    video_results()

elif selected == 'Upload Video':
    upload_video_view()

elif selected == "Analysis":
    analysis()
