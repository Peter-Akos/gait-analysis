from rest_framework import generics

from video_processing import process_video
from .models import Patient
from .serializers import PatientSerializer

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Video
from django.core.files.base import ContentFile


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


@csrf_exempt
def upload_video(request):
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        video_file = request.FILES.get('video_file')

        # Ensure the patient exists
        try:
            patient = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist:
            return JsonResponse({'error': 'Patient not found'}, status=404)

        # Compute the metrics
        metrics = process_video(video_file)

        # Create a new Video instance
        video = Video(
            patient=patient,
            speed_left=metrics['speed_left'],
            speed_right=metrics['speed_right'],
            cadence_left=metrics['cadence_left'],
            cadence_right=metrics['cadence_right'],
            knee_flexion_left=metrics['knee_flexion_left'],
            knee_flexion_right=metrics['knee_flexion_right'],
            video_file=ContentFile(video_file.read(), name=video_file.name)
        )

        # Save the Video instance
        video.save()

        return JsonResponse({'message': 'Video uploaded and metrics computed successfully'}, status=201)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)

