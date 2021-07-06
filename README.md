# ITSP-Speech-Emotion-Recognition-Dezipher
This repository store the speech emotion recognition web-app made as ITSP. This project was done by Team Technologic: [Vedang Asgaonkar](https://github.com/VedangAsgaonkar), [Prathamesh Pilkhane](https://github.com/Prathamesh2708), [Parshant Arora](https://github.com/Parshant-Arora) and [Pulkit Agarwal](https://github.com/PulkitAgr113).

## The models
[Training Models](Training%20Models) contains the Jupyter notebooks on which the models were trained. We made use of text data from [these files](https://drive.google.com/drive/folders/1x5m4OnLF-xZoMSz36ikw4ffc1SnRsfDe?usp=sharing). [Stanford GLoVe Embeddigs](https://github.com/stanfordnlp/GloVe) stored [in this folder](https://drive.google.com/file/d/1L0TcBVaWnOJt-TsT7GC-dQq4Lb59B-nA/view?usp=sharing) were used in the words based model. The frequency/ tone based model made use of [IEMOCAP data set](https://sail.usc.edu/iemocap/) stored in [this folder](https://drive.google.com/drive/folders/1eGqZ_gxJmm6Y7rc-gzbLHOjyT-vaARfq?usp=sharing).

The trained model weights are stored [here](https://drive.google.com/drive/folders/1xUz0hBP1nSk6Tf7jfpFy5CIUKf0_qTt5?usp=sharing).

## Webapp
[The Webapp](Webapp) Dezipher has been made to deploy the models. It has a Flask back-end and makes use of HTML, CSS and Javascript on the front-end. [Mattdiamond Recorderjs](https://github.com/mattdiamond/Recorderjs) has been used for taking live audio-input from the user. 

## Webapp Deployment
The Dezipher webapp will soon be deployed on Google App Engine

## Webapp Installation
Beside the online version, a user may also install the Dezipher web app and run it on their own machine. The steps of installation

### Creating Directory Structure
Clone/ download the webapp folder in this repository. In the [models](webapp/models) folder download the models from [here](https://drive.google.com/drive/folders/1xUz0hBP1nSk6Tf7jfpFy5CIUKf0_qTt5?usp=sharing). In the [text](webapp/text) folder download [this](https://drive.google.com/file/d/1vkggJKi-QPSA-tpROvZisW5KvdBLzc83/view?usp=sharing) and [this](https://drive.google.com/file/d/1hYjiT3RM8_l6sH2HlYsPwL5lnFHuKnq_/view?usp=sharing) file. In the [glove](webapp/glove) folder download [this](https://drive.google.com/file/d/14i7sZCaTE9wmOFdPEkvarVdTWtupff9N/view?usp=sharing) file.
```
webapp
|_downloads
|_glove
|_main.py
|_models
|_static
|_templates
|_text
```
