from django.db import models


class Patient(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField()
    gender_choices = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]
    gender = models.CharField(max_length=1, choices=gender_choices)
    address = models.TextField()
    phone_number = models.CharField(max_length=15)
    email = models.EmailField(unique=True)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


def video_directory_path(instance, filename):
    return f"patients_data/patient_{instance.patient.id}/videos/{filename}"


class Video(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='videos')
    speed_left = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    speed_right = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    cadence_left = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    cadence_right = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    video_file = models.FileField(upload_to=video_directory_path)
    processed = models.BooleanField(default=False)
    coordinates = models.JSONField(blank=True, null=True)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video {self.video_file.name} for {self.patient}"
