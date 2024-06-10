import subprocess
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from patient.models import Video

from model.models import CNN


def process_video(patient_id, filename):
    print(patient_id, filename)
    command = ["bin\\OpenPoseDemo.exe", "--video", f"..\\patients_data\\patient_{patient_id}\\videos\\{filename}",
               "--write_json",
               f"..\\patients_data\\patient_{patient_id}\\videos\\", "--display", "0", "--render_pose", "0"]

    current_directory = os.getcwd()

    # Specify the working directory
    working_directory = os.path.join(current_directory, "openpose")

    print(f"Started processing {filename} for patient with id {patient_id}")

    result = subprocess.run(command, cwd=working_directory, shell=True, capture_output=True)

    print(f"Finished processing {filename} for patient with id {patient_id}")

    filename_without_type = filename.split('.mp4')[0]

    process_json_results(patient_id, filename_without_type)


def process_json_results(patient_id, filename):
    result_array = read_json_results(patient_id, filename)

    body_parts = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
                  "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                  "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    columns = [f"{body_part}_{elem}" for body_part in body_parts for elem in ['x', 'y', 'prob']]

    df = pd.DataFrame(result_array, columns=columns)

    used_body_parts = ["LAnkle", "RAnkle", "LKnee", "RKnee", "LHip", "RHip", "LBigToe", "RBigToe"]
    used_features = [f"{body_part}_{elem}" for body_part in used_body_parts for elem in ['x', 'y']]

    filtered_df = df[used_features].replace(0, np.nan)

    filtered_df = filtered_df.interpolate(axis=1, limit_direction='both')

    features_json = filtered_df.to_json()
    data = filtered_df.to_numpy()

    data_shape = (500, 16)

    # Make sure the input to the model has the correct shape

    prepared_data = np.zeros((500, 16))
    num_rows_to_copy = min(data.shape[0], data_shape[0])
    prepared_data[:num_rows_to_copy, :] = data[:num_rows_to_copy, :]

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(prepared_data)

    model_input = torch.tensor(scaled_X, dtype=torch.float32).unsqueeze(0)
    cadence_path = "models/CNN_30_0.05048786286430754.pth"
    speed_path = "models/CNN_30_0.07537441999669288.pth"

    cadence_model = load_CNN_model(cadence_path)
    speed_model = load_CNN_model(speed_path)

    with torch.no_grad():
        cadence_model.eval()
        speed_model.eval()
        output_cadence = cadence_model(model_input)[0]
        output_speed = speed_model(model_input)[0]

    video_path = f"patients_data/patient_{patient_id}/videos/{filename}.mp4"

    video = Video.objects.get(video_file=video_path)
    video.speed_left = float(output_speed[0])
    video.speed_right = float(output_speed[1])
    video.cadence_left = float(output_cadence[0])
    video.cadence_right = float(output_cadence[1])
    video.coordinates = features_json
    video.processed = True
    video.save()

    print(output_speed)
    print(output_cadence)

    print(f"Finished processing video {filename} for patient {patient_id}")


def read_json_results(patient_id, filename):
    frames = os.listdir(f"patients_data\\patient_{patient_id}\\videos")
    frames = [frame for frame in frames if filename in frame and frame.endswith(".json")]
    res = []

    for frame in frames:
        json_path = f"patients_data/patient_{patient_id}/videos/{frame}"
        try:
            with open(json_path) as data_file:
                data = json.load(data_file)
            if len(data['people']) == 0:
                res.append([0 for _ in range(75)])
            else:
                person = data['people'][0]
                keypoints = person['pose_keypoints_2d']
                res.append(keypoints)
        except Exception as e:
            print(f"An unexpected error occurred with file {json_path}: {e}")

    result = np.array(res)
    return result


def load_CNN_model(path):
    state_dict = torch.load(path)
    # print(state_dict)
    # print(state_dict.keys())
    last_layer_dim = state_dict['fc1.weight'].shape[0]
    model = CNN(last_layer_dim=last_layer_dim)
    # model = CNN()
    model.load_state_dict(state_dict)
    return model

process_json_results(3, "063fe25d-f911-4c54-a5b7-5bb7fdd2eeac")
