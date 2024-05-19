from rest_framework import generics
from .models import Patient
from .serializers import PatientSerializer
from rest_framework.response import Response


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
