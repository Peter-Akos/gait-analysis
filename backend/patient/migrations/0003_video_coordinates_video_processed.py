# Generated by Django 5.0.6 on 2024-06-07 09:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('patient', '0002_video'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='coordinates',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='video',
            name='processed',
            field=models.BooleanField(default=False),
        ),
    ]
