import json
import time
from datetime import datetime

import numpy as np
import streamlit as st
import pandas as pd

from backend_requests import add_patient_request, delete_patient, get_patients, update_patient, send_video, get_videos, \
    delete_video, get_patients_videos
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

        patient_id = st.text_input("Patient ID to delete")
        if st.button("Delete Patient"):
            success = delete_patient(patient_id)
            if success:
                st.success("Patient Deleted Successfully")
                st.rerun()
            else:
                st.error("Failed to delete patient.")

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
    st.title("Select the Patient and upload a video.")

    patient_id = st.text_input("Patient ID")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        with open("input_video.mp4", 'wb') as f:
            f.write(uploaded_file.getvalue())
        filename = resize_video(640, 480)
        with open(filename, 'rb') as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)

        if st.button('Upload to Backend'):
            if not patient_id:
                st.warning('Please enter a Video ID.')
            else:
                with st.spinner('Uploading...'):
                    success = send_video(video_bytes, patient_id, filename)

                    if success:
                        st.success('Video uploaded successfully!')
                    else:
                        st.error('Failed to upload video.')


def video_results():
    st.title("View Video Results")
    videos = get_videos()
    data = []
    coordinates = []

    for video in videos:
        cpy = video
        coordinates.append(cpy['coordinates'])
        cpy.pop("coordinates")
        data.append(cpy)

    if videos:
        df = pd.DataFrame(data)
        df = df.set_index('id').drop(columns=['video_file'])
        st.dataframe(df, use_container_width=True)

        st.header("Detailed overview")
        video_id = st.text_input("ID of the video")
        if st.button("View"):
            st.header("Results")
            video_index = -1
            for i in range(len(videos)):
                if videos[i]['id'] == int(video_id):
                    video_index = i

            current_coordinates = coordinates[video_index]

            if current_coordinates is None:
                st.error("This video is yet to be processed")
            else:
                current_coordinates = json.loads(current_coordinates)
                coordinates_df = pd.DataFrame(current_coordinates)
                # st.dataframe(coordinates_df, use_container_width=True)

                st.subheader("Left Ankle Position")
                st.line_chart(coordinates_df[['LAnkle_x', 'LAnkle_y']])

                st.subheader("Right Ankle Position")
                st.line_chart(coordinates_df[['RAnkle_x', 'RAnkle_y']])

                st.subheader("Left Knee Position")
                st.line_chart(coordinates_df[['LKnee_x', 'LKnee_y']])

                st.subheader("Right Knee Position")
                st.line_chart(coordinates_df[['RKnee_x', 'RKnee_y']])

        st.header("Delete a video")
        video_id = st.text_input("Video ID to delete")
        if st.button("Delete Video"):
            success = delete_video(video_id)
            if success:
                st.success("Video Deleted Successfully")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to delete video.")


def analysis():
    st.title("Analysis of the progress of a patient")
    patient_id = st.text_input("Patient ID")
    if patient_id and st.button("See Results"):
        patients_videos = get_patients_videos(int(patient_id))
        if patients_videos:
            data = []
            videos = patients_videos['videos']

            for video in videos:
                data.append([
                    pd.to_datetime(video['date']),
                    video['speed_left'],
                    video['speed_right'],
                    video['cadence_left'],
                    video['cadence_right'],
                ])

            df = pd.DataFrame(data, columns=['date', 'speed_left', 'speed_right', 'cadence_left', 'cadence_right'])

            speed_left = np.random.uniform(0.6, 1, 10)
            speed_right = np.random.uniform(0.6, 1, 10)
            cadence_left = np.random.uniform(0.5, 1, 10)
            cadence_right = np.random.uniform(0.5, 1, 10)
            trend = np.linspace(0, 0.8, 10)
            date = pd.date_range('2023-10-3', periods=10, end='2024-05-19')

            df_random = pd.DataFrame()

            df_random['Speed - Left Side'] = speed_left + trend
            df_random['Speed - Right Side'] = speed_right + trend
            df_random['Cadence - Left Side'] = cadence_left + trend
            df_random['Cadence - Right Side'] = cadence_right + trend
            df_random['date'] = date

            st.dataframe(df_random)

            st.title("Speed over time")

            st.line_chart(df_random.set_index('date')[['Speed - Left Side', 'Speed - Right Side']])

            st.title("Cadence over time")
            st.line_chart(df_random.set_index('date')[['Cadence - Left Side', 'Cadence - Right Side']])


            # st.line_chart(df.set_index('date')[['random_speed_left', 'random_speed_right']])
            # st.line_chart(df.set_index('date')[['random_cadence_left', 'random_cadence_right']])
