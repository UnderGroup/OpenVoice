FROM python:3.10

WORKDIR /var/www/html

ADD requirements.txt .
RUN pip install -r requirements.txt

CMD ["python3.10", "main.py", "--console", "0"]
