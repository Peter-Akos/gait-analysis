import uuid
from io import BytesIO

import cv2
import numpy as np

from backend_requests import update_patient
import moviepy.editor as mp


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


def resize_video(width, height):
    clip = mp.VideoFileClip("input_video.mp4")
    clip_resized = clip.resize(newsize=(width, height))
    filename = f"{uuid.uuid4()}.mp4"
    clip_resized.write_videofile(filename)
    return filename


if __name__ == '__main__':
    resize_video(None, 640, 480)
