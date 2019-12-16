# ChatBot Model

## Environment Setup

### Setup with virtual environment (Recommended)

* You are recommended to run this project on a virtual environment. 

  Check [this](https://hackmd.io/@9BKtK41BQTmiTLq93uQ2aw/ByGBMmXcH) out if you have no idea about virtual environment

```
$ virtualenv env
$ . env/bin/activate
$ pip -r requirements.txt
```
### Setup on your own environment

```$ pip -r requirements.txt```



## Training

**Follow these steps to start training your chatbot model**

### Step 1: Data Preprocessing

```
$ python data_preprocess.py
```

* The training data I use for training is [Cornell Movie--Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). I only use **movie_conversations.txt** and **movie_lines.txt** (which are in data/) for training.
* Each line in the movie conversation will be transformed into pairs of conversation, and saved to **train_processed.pkl**.
* It will create 50000 conversation dialog pairs by default, change this by ```$ python3 data_preprocess.py -max [max_data]```
* **Please be patient**, this will take some time. 

### Step 2: Start Training

```
$ python train.py -c config.json -d 0
```

* Checkpoints of the trained model will be save to ```saved/model/[model_name]/[timestamp]```
* Logs  of the trained model will be save to ```saved/log/[model_name]/[timestamp]```

### Optional: Run Tensorboard Visualization

* Run the following command at the project root to start Tensorboard server.

```
$ tensorboard --logdir saved/log/[model_name]/[timestamp]
```

## Testing

> to be done.

### 