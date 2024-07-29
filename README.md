# TRAFFICLIGHT CLASSIFICATION 
## Create Environment and installation of packages
* Open Terminal in Linux
* conda create --name myenv python=3.9
* conda activate myenv
* pip install -r requirements.txt

## Execute Format 
* mkdir TrafficLightCLassification
* cd TrafficLightClassification
* gitclone 
* ### TRAINING
* python3 TrafficLightCLassification/train_weighted_sampler_red.py

## Training Dataset Format
The directory structure of the finally constructed training data set is as follows:
  ```
FINAL_DATASET_TRAFFIC_LIGHT  # train dataset
├── others                
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   └── .
│   └── .
│   └── .
├── red                                
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── .
│   └── .
│   └── .
```

## ML Flow Integration 
* mlflow ui --> to check logs

* Download the training and validation dataset from [Google Drive Link](https://drive.google.com/drive/folders/1NovbAleiGQYiy8Rj3PKlK_FV0JjL_yyv?usp=drive_link)
