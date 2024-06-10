from datetime import datetime

import streamlit as st
import pandas as pd

from backend_requests import add_patient_request, delete_patient, get_patients, update_patient, send_video
from utils import save_changed_data, resize_video


def add_patient():
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
            if add_patient_request(data):
                st.success("Patient added successfully!")
            else:
                st.error("Failed to add patient")
        else:
            st.error("All fields are required")


def view_patients():
    patients = get_patients()

    if patients:
        df = pd.DataFrame(patients)
        df = df.set_index('id')

        edited_data = st.data_editor(df, use_container_width=True, num_rows="fixed")

        # Check if data was modified
        if edited_data is not None and not df.equals(edited_data):
            if st.button('Save Changes'):
                success = save_changed_data(edited_data, df)
                if success:
                    st.success("Changes saved successfully!")
                else:
                    st.error("Failed to save changes")




        # for index, row in df.iterrows():
        #     st.write(f"### Patient ID: {index}")
        #     st.write(f"**First Name:** {row['first_name']}")
        #     st.write(f"**Last Name:** {row['last_name']}")
        #     st.write(f"**Date of Birth:** {row['date_of_birth']}")
        #     st.write(f"**Gender:** {row['gender']}")
        #     st.write(f"**Address:** {row['address']}")
        #     st.write(f"**Phone Number:** {row['phone_number']}")
        #     st.write(f"**Email:** {row['email']}")
        #
        #     with st.form(key=f"delete_form_{index}"):
        #         delete_button = st.form_submit_button(label="Delete")
        #         if delete_button:
        #             if delete_patient(index):
        #                 st.success(f"Patient {index} deleted successfully!")
        #                 st.experimental_rerun()
        #             else:
        #                 st.error(f"Failed to delete patient {index}")
        #     st.write("---")
    else:
        st.write("No patients found.")


def update_patient_view():
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
            if update_patient(patient_id_to_update, data):
                st.success("Patient updated successfully!")
            else:
                st.error("Failed to update patient")
        else:
            st.error("ID and all new details are required")


def upload_video_view():
    st.title("Select the Patient and upload a Video")

    patient_id = st.text_input("Patient ID")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file:
        with open("uploaded_file.mp4", 'wb') as f:
            f.write(uploaded_file.getvalue())
    print(uploaded_file)

    if uploaded_file is not None:
        resized_video_bytes = resize_video(uploaded_file.read(), 640, 480)
        st.video(uploaded_file, width=640, height=360)  # Display original video

        if st.button('Upload to Backend'):
            if not patient_id:
                st.warning('Please enter a Video ID.')
            else:
                with st.spinner('Uploading...'):
                    response = send_video(resized_video_bytes, patient_id)

                    if response.status_code == 200:
                        st.success('Video uploaded successfully!')
                    else:
                        st.error('Failed to upload video.')
