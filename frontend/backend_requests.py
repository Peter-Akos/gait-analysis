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
    return response.status_code == 200


def send_video(file, patient_id):
    backend_url = "http://localhost:5000/upload"  # Replace with your actual backend URL
    files = {'file': file}
    data = {'video_id': patient_id}
    response = requests.post(backend_url, files=files, data=data)
    return response.status_code == 200
