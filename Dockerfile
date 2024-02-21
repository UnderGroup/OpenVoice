FROM python:3.10

WORKDIR /var/www/html

RUN apt-get update -y && apt-get install -y ffmpeg

ADD requirements.txt .
RUN pip install -r requirements.txt

CMD ["python3.10", "web.py", "--console", "0"]
