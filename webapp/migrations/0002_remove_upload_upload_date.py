# Generated by Django 3.0.6 on 2020-06-05 09:43

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('webapp', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='upload',
            name='upload_date',
        ),
    ]
