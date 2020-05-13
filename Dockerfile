FROM  python:3.6

MAINTAINER Aaron W. Chen "awc33@cornell.edu"

RUN apt-get update -y && \ 
    apt-get install -y python-pip python-dev

COPY . /app

WORKDIR /app

RUN pip install -r docker_requirements.txt

COPY . /app

ENTRYPOINT ["python"]

CMD ["app.py"]