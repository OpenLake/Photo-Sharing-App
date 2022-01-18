from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import logout, login
from django.http import HttpResponse
from home.models import Photo, Person, PersonGallery
import string
import random


# Required for image processing
from imutils import paths
import face_recognition
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import re

# Required for downloading
import urllib3
from zipfile import ZipFile
from io import StringIO
import urllib.request
import os
import zipfile
from zipfile import ZipFile


post_type = re.compile(r"static/images/(.*)")


# Create your views here.
def index(request):
    user = request.user
    if user.is_anonymous:
        return redirect("/login")

    photos = Photo.objects.filter(user=user)
    count = photos.count()

    context = {"photos": photos, "count": count}
    return render(request, "index.html", context)


def loginUser(request):
    if request.method == "POST":
        roomcode = request.POST.get("roomCode")
        password = request.POST.get("inputPassword")
        user = authenticate(username=roomcode, password=password)
        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            return render(request, "login.html")
    return render(request, "login.html")


def logoutUser(request):
    logout(request)
    return redirect("/login")


def registerUser(request):
    global res
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    context = {"rcode": res}

    return render(request, "register.html", context)


def registerUser2(request):
    if request.method == "POST":
        roomcode = res
        password = request.POST.get("inputPassword")
        user = User.objects.create_user(roomcode, "", password)
        user.save()
        return redirect("/login")


def addPhoto(request):
    user = request.user

    if request.method == "POST":
        images = request.FILES.getlist("images")
        for image in images:
            print(image)
            photo = Photo.objects.create(user=user, image=image)
            photo.save()
        return redirect("index")
    return render(request, "add.html")


def viewPhoto(request, pk):
    photo = Photo.objects.get(id=pk)
    return render(request, "photo.html", {"photo": photo})


def process(request):
    user = request.user
    if user.is_anonymous:
        return redirect("/login")
    Person.objects.filter(user=user).delete()
    photos = Photo.objects.filter(user=user)
    if photos.count() == 0:
        return render(request, "404.html")
    imagePaths = [("static/images/" + str(photo.image)) for photo in photos]
    data = []

    for (i, imagePath) in enumerate(imagePaths):
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        print(imagePath)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # build a dictionary of the image path, bounding box location,
        # and facial encodings for the current image
        d = [
            {"imagePath": imagePath, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)
        ]
        data.extend(d)

    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    imgPath = [d["imagePath"] for d in data]
    # cluster the embeddings
    clt = DBSCAN(
        metric="euclidean", n_jobs=-1, min_samples=1
    )  # of parallel jobs to run (-1 will use all CPUs)
    clt.fit(encodings)
    # determine the total number of unique faces found in the dataset
    labelIDs = np.unique(clt.labels_)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])

    for labelID in labelIDs:
        idxs = np.where(clt.labels_ == labelID)[0]
        owner_pic = data[idxs[0]]["imagePath"]
        image = cv2.imread(owner_pic)
        (top, right, bottom, left) = data[idxs[0]]["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        face_img_path = f"{user.get_username()}_owner{labelID}.jpg"
        cv2.imwrite(f"static/images/{face_img_path}", face)
        person = Person.objects.create(user=user, thumbnail=face_img_path)
        person.save()

        for i in idxs:
            src_direc = data[i]["imagePath"]
            link = post_type.search(src_direc)
            personGallery = PersonGallery.objects.create(
                person=person, image=str(link.group(1))
            )
            personGallery.save()

    persons = Person.objects.filter(user=user)
    context = {"persons": persons, "faces": numUniqueFaces}
    return render(request, "process.html", context)


def albumGallery(request):
    user = request.user
    persons = Person.objects.filter(user=user)
    numUniqueFaces = persons.count()
    context = {"persons": persons, "faces": numUniqueFaces}
    return render(request, "process.html", context)


def viewAlbum(request, pk):
    person = Person.objects.get(id=pk)
    personGalleryphotos = PersonGallery.objects.filter(person=person)
    count = personGalleryphotos.count()
    allPhotos = personGalleryphotos.all()
    context = {
        "person": person,
        "personGalleryphotos": personGalleryphotos,
        "count": count,
    }
    return render(request, "personGallery.html", context)


def finalPhoto(request, pk):
    personPhoto = PersonGallery.objects.get(id=pk)
    context = {"personPhoto": personPhoto}
    return render(request, "finalPhoto.html", context)


from django.http import FileResponse

import tempfile, zipfile
from django.http import HttpResponse
from wsgiref.util import FileWrapper


def downloadZIP(request, pk):
    """
    Create a ZIP file on disk and transmit it in chunks of 8KB,
    without loading the whole file into memory. A similar approach can
    be used for large dynamic PDF files.
    """
    person = Person.objects.get(id=pk)
    personGalleryphotos = PersonGallery.objects.filter(person=person)

    allPhotos = personGalleryphotos.all()
    temp = tempfile.TemporaryFile()
    archive = zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED)
    for photo in allPhotos:
        print(os.getcwd())
        filename = str("/static") + photo.image.url  # Replace by your files here.

        archive.write(filename, "j.jpg")  # 'file%d.png' will be the
        # name of the file in the
        # zip
    archive.close()

    temp.seek(0)
    wrapper = FileWrapper(temp)

    response = HttpResponse(wrapper, content_type="application/zip")
    response["Content-Disposition"] = "attachment; filename=test.zip"

    return response
