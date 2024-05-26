import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import altair as alt

# Backend API URL
BASE_URL = "http://localhost:8000/api"


# Helper functions to interact with the API
def get_items():
    response = requests.get(f"{BASE_URL}/patients/")
    return response.json() if response.status_code == 200 else []


def create_item(data):
    response = requests.post(f"{BASE_URL}/patients/", json=data)
    return response.status_code == 201


def update_item(item_id, data):
    response = requests.put(f"{BASE_URL}/patients/update/{item_id}/", json=data)
    return response.status_code == 200


def delete_item(item_id):
    response = requests.delete(f"{BASE_URL}/patients/{item_id}/delete/")
    return response.status_code == 200


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


# Streamlit app
st.title("Patient Management System")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Add Patient", "View Patients", "Update Patient", "Delete Patient", "Visualize Results"],
        icons=["plus", "list", "pencil", "trash", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# Add Patient page
if selected == "Add Patient":
    st.header("Add a New Patient")
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    date_of_birth = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
    gender = st.selectbox("Gender", ["M", "F"])
    address = st.text_area("Address")
    phone_number = st.text_input("Phone Number")
    email = st.text_input("Email")

    if st.button("Add Patient"):
        if all([first_name, last_name, date_of_birth, gender, address, phone_number, email]):
            data = {
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth.isoformat(),
                "gender": gender,
                "address": address,
                "phone_number": phone_number,
                "email": email
            }
            if create_item(data):
                st.success("Patient added successfully!")
            else:
                st.error("Failed to add patient")
        else:
            st.error("All fields are required")

# View Patients page
elif selected == "View Patients":
    st.header("Patient List")
    patients = get_items()

    if patients:
        df = pd.DataFrame(patients)
        df = df.set_index('id')

        for index, row in df.iterrows():
            st.write(f"### Patient ID: {index}")
            st.write(f"**First Name:** {row['first_name']}")
            st.write(f"**Last Name:** {row['last_name']}")
            st.write(f"**Date of Birth:** {row['date_of_birth']}")
            st.write(f"**Gender:** {row['gender']}")
            st.write(f"**Address:** {row['address']}")
            st.write(f"**Phone Number:** {row['phone_number']}")
            st.write(f"**Email:** {row['email']}")

            with st.form(key=f"delete_form_{index}"):
                delete_button = st.form_submit_button(label="Delete")
                if delete_button:
                    if delete_item(index):
                        st.success(f"Patient {index} deleted successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete patient {index}")
            st.write("---")
    else:
        st.write("No patients found.")

# Update Patient page
elif selected == "Update Patient":
    st.header("Update Patient Details")
    patient_id_to_update = st.number_input("Enter the ID of the patient to update", min_value=1, step=1)
    new_first_name = st.text_input("New First Name")
    new_last_name = st.text_input("New Last Name")
    new_date_of_birth = st.date_input("New Date of Birth", min_value=datetime(1900, 1, 1))
    new_gender = st.selectbox("New Gender", ["M", "F"])
    new_address = st.text_area("New Address")
    new_phone_number = st.text_input("New Phone Number")
    new_email = st.text_input("New Email")

    if st.button("Update Patient"):
        if patient_id_to_update and all(
                [new_first_name, new_last_name, new_date_of_birth, new_gender, new_address, new_phone_number,
                 new_email]):
            data = {
                "first_name": new_first_name,
                "last_name": new_last_name,
                "date_of_birth": new_date_of_birth.isoformat(),
                "gender": new_gender,
                "address": new_address,
                "phone_number": new_phone_number,
                "email": new_email
            }
            if update_item(patient_id_to_update, data):
                st.success("Patient updated successfully!")
            else:
                st.error("Failed to update patient")
        else:
            st.error("ID and all new details are required")

# Delete Patient page
elif selected == "Delete Patient":
    st.header("Delete a Patient")
    patient_id_to_delete = st.number_input("Enter the ID of the patient to delete", min_value=1, step=1)

    if st.button("Delete Patient"):
        if patient_id_to_delete:
            if delete_item(patient_id_to_delete):
                st.success("Patient deleted successfully!")
            else:
                st.error("Failed to delete patient")
        else:
            st.error("ID is required")

# Visualize Results page
# Visualize Results page
elif selected == "Visualize Results":
    st.header("Visualize Patient Data")
    ankle_positions = get_ankle_positions()
    predicted_features = get_predicted_features()

    if ankle_positions:
        timestamps = [entry["timestamp"] for entry in ankle_positions]
        left_ankle_x = [entry["left_ankle"][0] for entry in ankle_positions]
        left_ankle_y = [entry["left_ankle"][1] for entry in ankle_positions]
        right_ankle_x = [entry["right_ankle"][0] for entry in ankle_positions]
        right_ankle_y = [entry["right_ankle"][1] for entry in ankle_positions]

        # Create DataFrame for plotting
        data = {
            'Timestamp': timestamps,
            'Left Ankle X': left_ankle_x,
            'Left Ankle Y': left_ankle_y,
            'Right Ankle X': right_ankle_x,
            'Right Ankle Y': right_ankle_y
        }
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        # Plotting with Streamlit's built-in line_chart
        st.write("## Left Ankle Position Over Time")
        st.line_chart(df.set_index('Timestamp')[['Left Ankle X', 'Left Ankle Y']], width=0, height=0)

        st.write("## Right Ankle Position Over Time")
        st.line_chart(df.set_index('Timestamp')[['Right Ankle X', 'Right Ankle Y']], width=0, height=0)
    else:
        st.write("No ankle position data available.")

    if predicted_features:
        predicted_data = {
            'Feature': ['Speed', 'Cadence', 'Knee Flexion'],
            'Value': [predicted_features['Speed'], predicted_features['Cadence'], predicted_features['Knee Flexion']]
        }
        df_predicted = pd.DataFrame(predicted_data)

        # Rotate x-axis labels by 90 degrees
        chart = alt.Chart(df_predicted).mark_bar().encode(
            x=alt.X('Feature', title='Feature', axis=alt.Axis(labelAngle=0)),
            y='Value'
        ).properties(
            width=500,
            height=300
        )

        # Display the bar chart
        st.write("## Predicted Features")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No predicted features data available.")
