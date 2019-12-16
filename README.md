# Task1-Tagging_of_Thesis

#### Install Environment

```
$ python3 -m venv competition
$ . ./competition/bin/activate
$ pip -r requirements.txt
```
### Word Embedding

https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip

### Run Tensorboard Visualization
Run below command at the project root, then server will open at `http://localhost:6006`
```
$ tensorboard --logdir saved/log/
```

### Data Preprocess

```
$ python ./data_preprocess.py --train ./data/task1_trainset.csv --test ./data/task1_public_testset.csv
```

### Training

```
$ python train.py -c config.json -d 0
```

### Testing

```
$ python test.py -c config.json -d 0 --resume saved/models/ThesisTagging/1010_013058/model_best.pth
```

