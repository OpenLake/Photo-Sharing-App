from django.urls import path
from home import views

urlpatterns = [
    path("", views.index, name="index"),
    path("landing", views.landing, name="landing"),
    path("login", views.loginUser, name="login"),
    path("logout", views.logoutUser, name="logout"),
    path("register", views.registerUser, name="register"),
    path("register2", views.registerUser2, name="register2"),
    path("photo/<str:pk>/", views.viewPhoto, name="photo"),
    path("delete/<str:pk>/", views.deletePhoto, name="delete"),
    path("album/<str:pk>/", views.viewAlbum, name="album"),
    path("finalPhoto/<str:pk>/", views.finalPhoto, name="finalPhoto"),
    path("download/<str:pk>/", views.downloadZIP, name="download"),
    path("add", views.addPhoto, name="add"),
    path("albumGallery", views.albumGallery, name="albumGallery"),
    path("process", views.process, name="process"),
]
