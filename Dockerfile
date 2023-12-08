FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=1 --bind 0.0.0.0:5000 app:app
#CMD gunicorn --workers=1 --bind 0.0.0.0:8080 app:app
