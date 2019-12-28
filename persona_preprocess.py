import os
from pprint import pprint

import pandas as pd
import pickle
import argparse
import logging
import math

import json

training_data_save_path = os.path.join("data", "train_processed.pkl")
testing_data_save_path = os.path.join("data", "test_processed.pkl")
training_data_path = os.path.join('data', 'summer_wild_evaluation_dialogs.json')
training_data_conversation_path = os.path.join('data', 'movie_conversations.txt')

def main(args):
    max_data = args.max
    logging.info('Process Training Data...')
    with open(training_data_path, 'r') as f:
        data = json.load(f)

    pairs = []

    for dialog in data:
        dialog = dialog['dialog']
        dialog = [line['text'] for line in dialog]
        for i in range(0, len(dialog)-1, 2):
            pairs.append([dialog[i], dialog[i+1]])


    logging.info("Number of training data: {}".format(len(pairs)))
    with open(training_data_save_path, "wb") as f:
        pickle.dump(pairs, f)
        logging.info("Processed training data save to {}".format(training_data_save_path))

    # line_df = pd.read_csv(training_data_path, sep=" \+\+\+\$\+\+\+ ", names=['lineID', 'characterID', 'movieID', 'character', 'utterance'], engine='python')
    # conversation_df = pd.read_csv(training_data_conversation_path, sep=" \+\+\+\$\+\+\+ ", names=['character1ID', 'character2ID', 'movieID', 'utterance_list'], engine='python')
    # #print(line_df.head(5))
    # #print(conversation_df.head(5))
    # #print(line_df.shape)
    # #print(conversation_df.shape)
    # utterance_list = conversation_df.loc[2]['utterance_list'].strip("[]'").split("', '")
    # #print(utterance_list)
    # # for lineID in utterance_list:
    # #     print(line_df[line_df['lineID'] == lineID])

    # processed_data = []
    # for index, conversation in conversation_df.iterrows():
    #     if(index >= max_data): break
    #     utterance_list = conversation['utterance_list'].strip("[]'").split("', '")
    #     #print(utterance_list)
    #     pair = [line_df[line_df['lineID'] == utterance_list[0]]['utterance'].item(), line_df[line_df['lineID'] == utterance_list[1]]['utterance'].item()]
    #     processed_data.append(pair)

    
    # logging.info("Number of training data: {}".format(len(processed_data)))
    # with open(training_data_save_path, "wb") as f:
    #     pickle.dump(processed_data, f)
    #     logging.info("Processed training data save to {}".format(training_data_save_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('--train', default=None, type=str,
                      help='path to training data (default: None)')
    parser.add_argument('--test', default=None, type=str,
                      help='path to testing data (default: None)')
    parser.add_argument('--max', default=50000, type=int,
                      help='maximum number of training data (default: None)')
    args = parser.parse_args()

    main(args)
