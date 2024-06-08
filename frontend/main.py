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

# Visualize Results page
# Visualize Results page
# elif selected == "Visualize Results":
#     st.header("Visualize Patient Data")
#     ankle_positions = get_ankle_positions()
#     predicted_features = get_predicted_features()
#
#     if ankle_positions:
#         timestamps = [entry["timestamp"] for entry in ankle_positions]
#         left_ankle_x = [entry["left_ankle"][0] for entry in ankle_positions]
#         left_ankle_y = [entry["left_ankle"][1] for entry in ankle_positions]
#         right_ankle_x = [entry["right_ankle"][0] for entry in ankle_positions]
#         right_ankle_y = [entry["right_ankle"][1] for entry in ankle_positions]
#
#         # Create DataFrame for plotting
#         data = {
#             'Timestamp': timestamps,
#             'Left Ankle X': left_ankle_x,
#             'Left Ankle Y': left_ankle_y,
#             'Right Ankle X': right_ankle_x,
#             'Right Ankle Y': right_ankle_y
#         }
#         df = pd.DataFrame(data)
#         df['Timestamp'] = pd.to_datetime(df['Timestamp'])
#
#         # Plotting with Streamlit's built-in line_chart
#         st.write("## Left Ankle Position Over Time")
#         st.line_chart(df.set_index('Timestamp')[['Left Ankle X', 'Left Ankle Y']], width=0, height=0)
#
#         st.write("## Right Ankle Position Over Time")
#         st.line_chart(df.set_index('Timestamp')[['Right Ankle X', 'Right Ankle Y']], width=0, height=0)
#     else:
#         st.write("No ankle position data available.")
#
#     if predicted_features:
#         predicted_data = {
#             'Feature': ['Speed', 'Cadence', 'Knee Flexion'],
#             'Value': [predicted_features['Speed'], predicted_features['Cadence'], predicted_features['Knee Flexion']]
#         }
#         df_predicted = pd.DataFrame(predicted_data)
#
#         # Rotate x-axis labels by 90 degrees
#         chart = alt.Chart(df_predicted).mark_bar().encode(
#             x=alt.X('Feature', title='Feature', axis=alt.Axis(labelAngle=0)),
#             y='Value'
#         ).properties(
#             width=500,
#             height=300
#         )
#
#         # Display the bar chart
#         st.write("## Predicted Features")
#         st.altair_chart(chart, use_container_width=True)
#     else:
#         st.write("No predicted features data available.")

elif selected == 'Upload Video':
    upload_video_view()

elif selected == "Analysis":
    analysis()
