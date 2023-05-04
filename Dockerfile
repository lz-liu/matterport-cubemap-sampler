FROM python:3.8-slim-bullseye

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    xvfb

# Set up a virtual display using Xvfb
ENV DISPLAY=:99

WORKDIR /scripts

COPY reproj_fbo.py /scripts

RUN pip install PyOpenGL numpy Pillow

ENTRYPOINT ["sh", "-c", "nohup Xvfb :99 -ac & export DISPLAY=:99 && python reproj_fbo.py \"$@\"", "--"]
