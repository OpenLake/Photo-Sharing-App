#!/bin/bash

python3 backend/manage.py makemigrations
python3 backend/manage.py migrate
python3 backend/manage.py runserver ${1}
