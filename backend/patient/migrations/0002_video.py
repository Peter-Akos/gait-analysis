# Generated by Django 4.2.13 on 2024-06-01 14:52

from django.db import migrations, models
import django.db.models.deletion
import patient.models


class Migration(migrations.Migration):

    dependencies = [
        ('patient', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('speed_left', models.DecimalField(decimal_places=2, max_digits=5)),
                ('speed_right', models.DecimalField(decimal_places=2, max_digits=5)),
                ('cadence_left', models.DecimalField(decimal_places=2, max_digits=5)),
                ('cadence_right', models.DecimalField(decimal_places=2, max_digits=5)),
                ('knee_flexion_left', models.DecimalField(decimal_places=2, max_digits=5)),
                ('knee_flexion_right', models.DecimalField(decimal_places=2, max_digits=5)),
                ('video_file', models.FileField(upload_to=patient.models.video_directory_path)),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='videos', to='patient.patient')),
            ],
        ),
    ]
