from ctypes.wintypes import HACCEL
from curses.ascii import HT
from http.client import HTTPResponse
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.contrib.auth import logout, login
from home.models import Photo
import string
import random

# Required for image processing
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import os
import shutil


# Create your views here.
def index(request):
    user = request.user
    if user.is_anonymous:
        return redirect("/login")

    photos = Photo.objects.filter(user=user)
    context = {"photos": photos}
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
        print("uer", user)
        user.save()
        return redirect("/login")


def addPhoto(request):
    user = request.user
    user = user.get_username()

    if request.method == "POST":
        images = request.FILES.getlist("images")

        for image in images:
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
    photos = Photo.objects.filter(user=user)

    imagePaths = [str(photo.image) for photo in photos]
    print(imagePaths)
    user2 = request.user
    photos = Photo.objects.filter(user=user2)
    context = {"photos": photos}
    return render(request, "process.html", context)
