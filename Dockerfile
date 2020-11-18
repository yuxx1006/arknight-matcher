FROM python:3.7

ENV MONGO 127.0.0.1:27017/arknights

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5005

CMD [ "python", "main.py" ]


