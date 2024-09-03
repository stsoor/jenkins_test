# Jenkins

## Dockerfile (Jenkins)

Base jenkins image but with docker installed for docker in docker usecases

## Dockerfile (GroundingDINO)

Custom docker for GD which has ffmpeg & codecs for opencv (possibly unnecessary), fixes for the original docker_test.py and added script for inference on videos.

## Jenkinsfile

Contains the pipeline for a docker in docker setup (except I couldn't share the GPUs again). Should be refit for a physical host.

## Useful code snippets

Connect to container as su:
```
bash
docker exec -it -u root <container> /bin/bash
```
Run vanilla jenkins docker:
```
bash
docker run -p 8080:8080 -p 50000:50000 --gpus all --restart=on-failure -v <volume name>:/var/jenkins_home jenkins/jenkins:lts-jdk17
```
Run docker-in-docker jenkins and clean up afterwards:
```
bash
docker run --rm -p 8080:8080 -p 50000:50000 --gpus all -v <volume name>:/var/jenkins_home --name <container name> <jenkins (with docker) image name>
```
Run docker-in-docker jenkins with shared windows sockets (unsafe):
```
bash
docker run --rm -p 8080:8080 -p 50000:50000 --gpus all -v <volume name>:/var/jenkins_home -e DOCKER_HOST=tcp://host.docker.internal:3872 --name <container name> <jenkins (with docker) image name>
-e DOCKER_HOST=tcp://host.docker.internal:3872
```
Share docker sockets under linux:
```
-v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker
```
```
bash
docker run -p 8080:8080 -p 50000:50000 --gpus all --restart=on-failure --privileged -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v <volume name>:/var/jenkins_home jenkins/jenkins:lts-jdk17
```
Create volume:
```
bash
docker volume create <volume_name>
```
Docker volume location under windows:
```
\\wsl.localhost\docker-desktop-data\data\docker\volumes\<volume_name>\_data
```
Docker volume under linux:
```
/var/lib/docker/volume/
```
Docker build command (in case of emergencies):
```
bash
docker build -t <image name> .
```

Example call for the inference on video:
```
python
inference_on_a_video.py -i /data/test.mp4 -o /out/ -t person -bt 0.4 -tt 0.25
```