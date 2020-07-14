from django.contrib import admin
from webapp.models import Upload

# Register your models here.


@admin.register(Upload)
class UploadAdmin(admin.ModelAdmin):
    pass
