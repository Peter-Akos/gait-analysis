import requests

from constants import BASE_URL


def add_patient_request(data):
    response = requests.post(f"{BASE_URL}/patients/", json=data)
    return response.status_code == 201


def get_patients():
    response = requests.get(f"{BASE_URL}/patients/")
    return response.json() if response.status_code == 200 else []


def update_patient(item_id, data):
    response = requests.put(f"{BASE_URL}/patients/update/{item_id}/", json=data)
    return response.status_code == 200


def delete_patient(item_id):
    response = requests.delete(f"{BASE_URL}/patients/{item_id}/delete/")
    return response.status_code == 204


def delete_video(item_id):
    response = requests.delete(f"{BASE_URL}/videos/{item_id}/delete/")
    return response.status_code == 204


def send_video(file, patient_id, filename):
    files = {'video_file': file}
    data = {'patient_id': patient_id, "filename": filename}
    response = requests.post(f"{BASE_URL}/upload/", files=files, data=data)
    return response.status_code == 201


def get_videos():
    response = requests.get(f"{BASE_URL}/videos/")
    return response.json() if response.status_code == 200 else []


def get_patients_videos(patient_id):
    response = requests.get(f"{BASE_URL}/patients/{patient_id}/videos/")
    return response.json() if response.status_code == 200 else []
