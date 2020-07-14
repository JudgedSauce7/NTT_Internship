from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect
from webapp.clean import doc

# Create your views here.


def index(request):
    upload = None
    processed = None
    fs = FileSystemStorage()
    fs.delete("uploaded.jpg")
    if request.method == "POST":
        upload = request.FILES['uploaded']
        # print(upload.name)
        fs = FileSystemStorage()
        fs.delete("uploaded.jpg")
        fs.save("uploaded.jpg", upload)
        processed = doc("/uploaded.jpg")
    return render(request, "index.html", {"upload": upload, "processed": processed})
