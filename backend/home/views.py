from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import logout, login
from django.http import HttpResponse
from home.models import Photo, Person, PersonGallery
import string
import random
from utils import checkGPUavailable, download_weights
import hdbscan
from django.contrib import messages


# Required for image processing
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import re

# Required for downloading
import os
import zipfile
import tempfile, zipfile
from django.http import HttpResponse, JsonResponse
from wsgiref.util import FileWrapper

# for checking face simmilarity
import scipy.spatial.distance as dist



# ReGEx required for getting photo name
post_type = re.compile(r"static/images/(.*)")


# Importing ML Library
import face_detection_embedding

# Create your views here.
# def landing(request):
#     return render(request, "landing.html")


def index(request):    
    user = request.user
    if user.is_anonymous:
        return redirect("/login")

    elif request.method == "POST":
        images = request.FILES.getlist("input-folder")
        for image in images:
            photo = Photo.objects.create(user=user, image=image)
            photo.save()
        Person.objects.filter(user=user).delete()
        photos = Photo.objects.filter(user=user)
        # imagePaths = [("static/images/" + str(photo.image)) for photo in photos]
        imagePaths = "static/images/"+ str(user)
        # print(request.method)
        models = request.POST["models"]
        # Yha se code start kro
        if models == "mediapipe":
            dic, unclassified = face_detection_embedding.detectPeopleMediaPipe({"path": imagePaths, "confidence": 0.25})
        elif models == "yolov7":
            # Check if weights exists if not then I will download them from internet
            download_weights("ML/yolov7face/weights/yolov7-face.pt", "https://drive.google.com/file/d/1k7zcq5_Vj8oqnkvll5Uvg57StOOYLnKn/view?usp=share_link")
            dic, unclassified = face_detection_embedding.detectPeopleYolov7({"weights":"ML/yolov7face/weights/yolov7-face.pt",
                                                                            "source": imagePaths,
                                                                            "img_size": 640,
                                                                            "conf_thres": 0.25,
                                                                            "iou_thres": 0.45,
                                                                            "device": '',
                                                                            "agnostic_nms":False,
                                                                            "classes": None,
                                                                            "augment": False,
                                                                            "hide_conf": False,
                                                                            "kpt_label": 5})
        else:
            # Check if weights exists if not then I will download them from internet
            download_weights("ML/yolov7face/weights/yolov7-w6-face.pt", "https://drive.google.com/file/d/1ThpUTdnEFG13WhGEPCI0Ut2NJYdq59-B/view?usp=share_link")
            dic, unclassified = face_detection_embedding.detectPeopleYolov7({"weights":"ML/yolov7face/weights/yolov7-w6-face.pt",
                                                                            "source": imagePaths,
                                                                            "img_size": 640,
                                                                            "conf_thres": 0.25,
                                                                            "iou_thres": 0.45,
                                                                            "device": '',
                                                                            "agnostic_nms":False,
                                                                            "classes": None,
                                                                            "augment": False,
                                                                            "hide_conf": False,
                                                                            "kpt_label": 5})                  
        
        # Checking to see if VGGFace weights exists
        download_weights("ML/FaceNet/Weights/facenet_keras_weights.h5", "https://drive.google.com/file/d/1QAYt7g9ig6UHYdpd1WW7P1Ci2XN32Kfd/view?usp=sharing")
        encodings = face_detection_embedding.generateEmbedding(dic)
        
        data = []
        for path, boundingboxes in dic.items():
            data.extend([{"imagePath": path, "loc": i,} for i in boundingboxes])
        
        print("Shape of encodings", np.array(encodings).shape)

        labels_ = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=int(request.POST["Size"])).fit_predict(encodings)
        labelIDs = np.unique(labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        
        for labelID in labelIDs:
            idxs = np.where(labels_ == labelID)[0]
            if labelID != -1:
                owner_pic = data[idxs[0]]["imagePath"]
                image = cv2.imread(owner_pic)
                (x1, y1, x2, y2) = data[idxs[0]]["loc"]
                face = cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
            else:
                face = cv2.imread("static/unclassified.jpg")

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
        nothing = {'nothing' : True}
        return JsonResponse(nothing)
    photos = Photo.objects.filter(user=user)
    count = photos.count()
    is_gpu = checkGPUavailable()
    context = {"photos": photos, "count": count, "GPU": is_gpu}
    return render(request, "index.html", context)
# def index(request):    
#     user = request.user
#     if user.is_anonymous:
#         return redirect("/login")

#     elif request.method == "POST":
#         images = request.FILES.getlist("images")
#         for image in images:
#             photo = Photo.objects.create(user=user, image=image)
#             photo.save()

#     photos = Photo.objects.filter(user=user)
#     count = photos.count()
#     is_gpu = checkGPUavailable()
#     context = {"photos": photos, "count": count, "GPU": is_gpu}
#     return render(request, "index.html", context)


def loginUser(request):
    if request.method == "POST" and "signup" in request.POST:
        first_name = request.POST.get("first_name")
        username = request.POST.get("username")
        password = request.POST.get("password")
        if User.objects.filter(username = username).first():
            alert = {"error": "Username already  in use."}
            return render(request, "login.html", alert)
            
        else:
            user = User.objects.create_user(username=username, password=password, first_name=first_name)
            user.save()
            return redirect("/login")
    if request.method == "POST" and "login" in request.POST:
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("/")
        else:
            messages.error(request, "Invalid Credentials")
            return render(request, "login.html")
    return render(request, "login.html")


def logoutUser(request):
    logout(request)
    return redirect("/login")


# def registerUser(request):
#     global res
#     res = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
#     context = {"rcode": res}

#     return render(request, "register.html", context)


# def registerUser2(request):
#     if request.method == "POST":
#         roomcode = res
#         password = request.POST.get("inputPassword")
#         user = User.objects.create_user(roomcode, "", password)
#         user.save()
#         return redirect("/login")


def viewPhoto(request, pk):
    photo = Photo.objects.get(id=pk)
    return render(request, "photo.html", {"photo": photo})

def images(request):
    user = request.user
    photos = Photo.objects.filter(user=user)
    count = photos.count()
    is_gpu = checkGPUavailable()
    context = {"photos": photos, "count": count, "GPU": is_gpu}
    return render(request, "images.html", context)

def deletePhoto(request, pk):
    if request.method == "POST":
        user = request.user
        photos = Photo.objects.filter(user=user, id=pk).values()
        photosa = Photo.objects.filter(user=user, id=pk)
        img = photos[0]
        photosa.delete()
        os.remove("static/images/" + img['image'])
    return redirect("images")


def album(request):
    user = request.user
    if user.is_anonymous:
        return redirect("/login")
    # Person.objects.filter(user=user).delete()
    photos = Photo.objects.filter(user=user)
    if photos.count() == 0:
        context = {
            "error_message": "No photos to process.\n Upload some photos and then Try again"
        }
        return render(request, "404.html", context)
    
    # if request.method != 'POST':
    #     return redirect("/login")
    
    # # imagePaths = [("static/images/" + str(photo.image)) for photo in photos]
    # imagePaths = "static/images/"+ str(user)
    # # print(request.method)
    # models = request.POST["models"]
    # # Yha se code start kro
    # if models == "mediapipe":
    #     dic, unclassified = face_detection_embedding.detectPeopleMediaPipe({"path": imagePaths, "confidence": 0.25})
    # elif models == "yolov7":
    #     # Check if weights exists if not then I will download them from internet
    #     download_weights("ML/yolov7face/weights/yolov7-face.pt", "https://drive.google.com/file/d/1k7zcq5_Vj8oqnkvll5Uvg57StOOYLnKn/view?usp=share_link")
    #     dic, unclassified = face_detection_embedding.detectPeopleYolov7({"weights":"ML/yolov7face/weights/yolov7-face.pt",
    #                                                                     "source": imagePaths,
    #                                                                     "img_size": 640,
    #                                                                     "conf_thres": 0.25,
    #                                                                     "iou_thres": 0.45,
    #                                                                     "device": '',
    #                                                                     "agnostic_nms":False,
    #                                                                     "classes": None,
    #                                                                     "augment": False,
    #                                                                     "hide_conf": False,
    #                                                                     "kpt_label": 5})
    # else:
    #     # Check if weights exists if not then I will download them from internet
    #     download_weights("ML/yolov7face/weights/yolov7-w6-face.pt", "https://drive.google.com/file/d/1ThpUTdnEFG13WhGEPCI0Ut2NJYdq59-B/view?usp=share_link")
    #     dic, unclassified = face_detection_embedding.detectPeopleYolov7({"weights":"ML/yolov7face/weights/yolov7-w6-face.pt",
    #                                                                     "source": imagePaths,
    #                                                                     "img_size": 640,
    #                                                                     "conf_thres": 0.25,
    #                                                                     "iou_thres": 0.45,
    #                                                                     "device": '',
    #                                                                     "agnostic_nms":False,
    #                                                                     "classes": None,
    #                                                                     "augment": False,
    #                                                                     "hide_conf": False,
    #                                                                     "kpt_label": 5})                  
    
    # # Checking to see if VGGFace weights exists
    # download_weights("ML/FaceNet/Weights/facenet_keras_weights.h5", "https://drive.google.com/file/d/1QAYt7g9ig6UHYdpd1WW7P1Ci2XN32Kfd/view?usp=sharing")
    # encodings = face_detection_embedding.generateEmbedding(dic)
    
    # data = []
    # for path, boundingboxes in dic.items():
    #     data.extend([{"imagePath": path, "loc": i,} for i in boundingboxes])
    
    # print("Shape of encodings", np.array(encodings).shape)

    # labels_ = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=int(request.POST["Size"])).fit_predict(encodings)
    # labelIDs = np.unique(labels_)
    # numUniqueFaces = len(np.where(labelIDs > -1)[0])
    
    # for labelID in labelIDs:
    #     idxs = np.where(labels_ == labelID)[0]
    #     if labelID != -1:
    #         owner_pic = data[idxs[0]]["imagePath"]
    #         image = cv2.imread(owner_pic)
    #         (x1, y1, x2, y2) = data[idxs[0]]["loc"]
    #         face = cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
    #     else:
    #         face = cv2.imread("static/unclassified.jpg")

    #     face_img_path = f"{user.get_username()}_owner{labelID}.jpg"
    #     cv2.imwrite(f"static/images/{face_img_path}", face)
    #     person = Person.objects.create(user=user, thumbnail=face_img_path)
    #     person.save()

    #     for i in idxs:
    #         src_direc = data[i]["imagePath"]
    #         link = post_type.search(src_direc)
    #         personGallery = PersonGallery.objects.create(
    #             person=person, image=str(link.group(1))
    #         )
    #         personGallery.save()
    user = request.user
    persons = Person.objects.filter(user=user)

    # score_list = []
    numUniqueFaces = persons.count()

    # result = []
    # for ele in range(numUniqueFaces):
    #     result.append(persons[ele])

    context = {"persons": persons, "faces": numUniqueFaces}
    return render(request, "process.html", context)


def albumGallery(request):
    user = request.user
    persons = Person.objects.filter(user=user)

    score_list = []
    numUniqueFaces = persons.count()
    path = os.getcwd() + str("/static/images/")

    for i in range(numUniqueFaces):
        intmd_lst = []
        for j in range(i + 1, numUniqueFaces):
            image1 = cv2.imread(path + str(persons[i].thumbnail))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2 = cv2.imread(path + str(persons[j].thumbnail))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            score = dist.cosine(image1.reshape(-1), image2.reshape(-1))
            intmd_lst.append([i, j, score])
            print(i, j, score)
        score_list.append(intmd_lst)

    result = []
    unsorted_person = [i for i in range(1, numUniqueFaces)]
    sorted_person = []
    start = 0
    next = score_list[0][0][1]
    sorted_person.append(start)
    i = 0
    for _ in range(numUniqueFaces - 1):
        print(score_list[start])
        min = 1
        for ele in score_list[start]:
            if ele[2] < min:
                min = ele[2]
                next = ele[1]
        print(min, next)

        if start == next:
            break
        else:
            unsorted_person.remove(next)
            sorted_person.append(next)
            start = next
    final = sorted_person + unsorted_person
    for ele in final:
        result.append(persons[ele])

    context = {"persons": result, "faces": numUniqueFaces}
    return render(request, "process.html", context)


def viewAlbum(request, pk):
    person = Person.objects.get(id=pk)
    personGalleryphotos = PersonGallery.objects.filter(person=person)
    count = personGalleryphotos.count()
    context = {
        "person": person,
        "personGalleryphotos": personGalleryphotos,
        "count": count,
    }
    return render(request, "personGallery1.html", context)


def finalPhoto(request, pk):
    personPhoto = PersonGallery.objects.get(id=pk)
    context = {"personPhoto": personPhoto}
    return render(request, "finalPhoto.html", context)


def downloadZIP(request, pk):
    person = Person.objects.get(id=pk)
    personGalleryphotos = PersonGallery.objects.filter(person=person)

    allPhotos = personGalleryphotos.all()
    temp = tempfile.TemporaryFile()
    archive = zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED)
    for photo in allPhotos:
        filename = (
            os.getcwd() + str("/static/images/") + photo.image.name
        )  # Replace by your files here.
        photo_name = photo.image.url[1:]
        archive.write(filename, f"{photo_name}")
    archive.close()
    temp.seek(0)
    wrapper = FileWrapper(temp)
    response = HttpResponse(wrapper, content_type="application/zip")
    response["Content-Disposition"] = "attachment; filename=album.zip"

    return response
