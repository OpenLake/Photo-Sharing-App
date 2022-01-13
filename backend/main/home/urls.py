from django.urls import path
from home import views

urlpatterns = [
    path('',views.index,name="index"),
    path('login', views.loginUser, name="login"),
    path('logout', views.logoutUser, name="logout"),
    path('register', views.registerUser, name="register"),
    path('register2', views.registerUser2, name="register2"),
    path('photo/<str:pk>/', views.viewPhoto, name='photo'),
    path('add',views.addPhoto,name="add"),
    path('process',views.process,name="process"),
]