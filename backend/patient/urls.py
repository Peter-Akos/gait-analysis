from django.urls import path
from .views import PatientListCreateView, PatientDetailView, PatientUpdateView, PatientDeleteView, upload_video, \
    VideoListCreateView, VideoDeleteView, patient_videos

urlpatterns = [
    path('patients/', PatientListCreateView.as_view(), name='patient-list-create'),
    path('patients/<int:pk>/', PatientDetailView.as_view(), name='patient-detail'),
    path('patients/update/<int:pk>/', PatientUpdateView.as_view(), name='patient-update'),
    path('patients/<int:pk>/delete/', PatientDeleteView.as_view(), name='patient-delete'),
    path('upload/', upload_video, name='upload_video'),
    path('videos/', VideoListCreateView.as_view(), name='video-list-create'),
    path('videos/<int:pk>/delete/', VideoDeleteView.as_view(), name='video-delete'),
    path('patients/<int:patient_id>/videos/', patient_videos, name='patient_videos'),
]
