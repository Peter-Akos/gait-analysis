from io import BytesIO

import cv2
import numpy as np

from backend_requests import update_patient


def save_changed_data(new_df, old_df):
    for index, row in new_df.iterrows():
        old_row = old_df.loc[index]
        if not row.equals(old_row):
            data = {
                "first_name": row['first_name'],
                "last_name": row['last_name'],
                "date_of_birth": row['date_of_birth'],
                "gender": row['gender'],
                "address": row['address'],
                "phone_number": row['phone_number'],
                "email": row['email']
            }
            if update_patient(index, data):
                print(f"Patient {index} updated successfully!")
            else:
                print(f"Failed to update patient {index}")
                return False
    return True


def resize_video(video_bytes, width, height):
    # video_array = np.frombuffer(video_bytes, dtype=np.uint8)
    # # video_array = np.load('data.npy')  # load
    # print(video_array.shape)
    # video = cv2.imdecode(video_array, 0)
    # print(video)
    #
    video = cv2.VideoCapture("sample_video.mp4")

    # Check if video was read successfully
    if video is None:
        raise ValueError("Unable to decode the video.")

    # Resize the video
    resized_video = cv2.resize(video, (width, height))

    # Encode resized video to bytes
    _, encoded_image = cv2.imencode('.mp4', resized_video)
    if not _:
        raise ValueError("Unable to encode the resized video.")

    resized_video_bytes = encoded_image.tobytes()

    return resized_video_bytes


if __name__ == '__main__':
    resize_video(None, 640, 480)
