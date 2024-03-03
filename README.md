# Photo-Sharing-App

## About the Project
Photo Sharing App is a web app that allows a user to get group photos that include them without having to keep or sort through all of the photos, saving them time and
storage space. When a photo is uploaded, the app identifies the faces in it and arranges them according to the users who have been identified. Each identified user gets
a separate album with all their photos allowing them to get the photos they need seamlessly.
Upload group photos, arrange them by faces using ML and share them with ease. 🤳🤖

## Installation
- Option 1: Use the deployed version at: https://photo-organizer.herokuapp.com/
-----------------------------------------------------
- Option 2: 
    1. ### Setting up git:
    - [Download and install the latest version of Git.](https://git-scm.com/downloads)
    - (Optional) [Set your username in Git.](https://help.github.com/articles/setting-your-username-in-git)
    - (Optional) [Set your commit email address in Git.](https://help.github.com/articles/setting-your-commit-email-address-in-git)
    2. ### Cloning the Repo in your local machine
    - Run ``git clone git@github.com:OpenLake/Photo-Sharing-App.git``
    - ``cd Photo-Sharing-App``
    - Make a virtual environment named .env ``python3 -m venv .env``
    - Activate ``.env`` by ``source .env/bin/activate``
    - Download the requirements ``pip install -r requirements.txt``
    - Run ``cd backend``
    - Run ``python3 manage.py makemigrations``
    - Run ``python3 manage.py migrate``
    - Run ``python3 manage.py runserver 8000``
    - Go to  http://127.0.0.1:8000/
    - You are good to go! 🤘

## Usage
![1](https://user-images.githubusercontent.com/72318258/150670154-05acfa34-7ffd-4bcb-a790-219501713454.png)
Landing Page


![2](https://user-images.githubusercontent.com/72318258/150670156-99289a1b-0606-4f3f-b438-82596afbc1b8.png)
Room Creation Page


![3](https://user-images.githubusercontent.com/72318258/150670158-1e18c160-9a04-4412-999a-051e228887cc.png)
Here I have uploaded 6 images of Harry Potter cast.


![4](https://user-images.githubusercontent.com/72318258/150670161-1a04b223-aa9a-4b64-a66d-822e5384fddf.png)
After pressing "Process" we get these 3 albums of faces present in the photos we uploaded.


![5](https://user-images.githubusercontent.com/72318258/150670162-4a8ae9d4-34a3-4c25-8e64-8442c0cd36ba.png)
This is Ron Weasley a.k.a Rupert Grint's album. Out of the original 6 images he was present in just 4 and now we can download these pics indiviually or as a zip.

## How does it work?

When a user uploads a group photos and starts to "Process" it, the [Face Recognition](https://github.com/ageitgey/face_recognition) library detects all the faces present in the group image. After detecting the faces it reduces them into a 128 length vector. Now each image is represented by a vector of length 128. Once we have obtained the vectors the task that remains is to cluster them. This being an unsupervised clustering problem so I have used [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html). This method doesnt need to know the number of clusters beforehand. It divides all the images into clusters where each cluster belongs to a single person as the simmilarity between the images of the same person is more hence they get clustered closely. Now each cluster gets an indiviual album from where you can download the album or a single image indiviually.


## Frequently Asked Questions

**Q: I got Error: That port is already in use.**

A: It occurs because some process has occupied that port. You may kill the process using ``sudo fuser -k 8000/tcp`` or use a different port. For using a different port simply replace ``bash setup.sh`` with ``bash setup.sh <port-number>``. An example ``bash setup.sh 7000``. Now your app will successfully run on http://127.0.0.1:7000/


