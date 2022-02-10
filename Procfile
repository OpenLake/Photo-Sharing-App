web: gunicorn core.wsgi:application --log-file - --log-level debug python backend/manage.py runserver 0.0.0.0:$PORT collectstatic --noinput backend/manage.py migrate
