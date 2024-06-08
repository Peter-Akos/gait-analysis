from django.shortcuts import get_object_or_404
from rest_framework import generics

from video_processing import process_video
from .models import Patient
from .serializers import PatientSerializer, VideoSerializer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Video
from django.core.files.base import ContentFile
import threading


class PatientListCreateView(generics.ListCreateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


class PatientUpdateView(generics.RetrieveUpdateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    partial = True


class PatientDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


class PatientDeleteView(generics.DestroyAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


class VideoListCreateView(generics.ListCreateAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer


class VideoDeleteView(generics.DestroyAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer


@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        video_filename = request.POST.get('filename')
        video_file = request.FILES.get('video_file')
        print(video_file)
        print(type(video_file))


        # Ensure the patient exists
        try:
            patient = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist:
            return JsonResponse({'error': 'Patient not found'}, status=404)

        thread = threading.Thread(target=process_video, args=(patient_id, video_filename))
        thread.start()

        # Create a new Video instance
        video = Video(
            patient=patient,
            video_file=ContentFile(video_file.read(), name=video_filename),
            processed=False
        )

        # Save the Video instance
        video.save()

        return JsonResponse({'message': 'Video uploaded and metrics computed successfully'}, status=201)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def patient_videos(request, patient_id):
    # Get the patient object
    patient = get_object_or_404(Patient, id=patient_id)

    # Get all videos related to the patient
    videos = patient.videos.order_by('id')

    # Serialize video data
    video_data = []
    for video in videos:
        if video.processed:
            video_data.append({
                'id': video.id,
                'date': video.date_created,
                'speed_left': video.speed_left,
                'speed_right': video.speed_right,
                'cadence_left': video.cadence_left,
                'cadence_right': video.cadence_right,
            })

    # Return JSON response
    return JsonResponse({'videos': video_data})
