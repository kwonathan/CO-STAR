# import necessary files and modules

from colours import col
from dataset_gen import DatasetGen

import json
import torch
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel
from transformers import GPT2Config



# file paths

DATA_TRN_PATH = './data/full_data_approved.concat.v2.csv'
DATA_DEV_PATH = './data/SBIC.v2/SBIC.v2.dev.csv'
DATA_TST_PATH = './data/SBIC.v2/SBIC.v2.tst.csv'

MODEL_SAVE_PATH = ''

# note that GPT2TokenizerFast throws an error when processing these words
# use GPT2Tokenizer instead
BADWORDS_PATH = './badwords.json'



# model hyperparameters

LEARNING_RATE = 1e-5
TRAIN_EPOCH = 5
BATCH_SIZE = 1
SUBSET = False
TESTING = False

##### OPTIONS ######################################################
# GPT_MODEL = 'gpt2' or 'gpt2-medium' or 'gpt2-large' or 'gpt2-xl' #
# GEN_TYPE = 'beam' or 'topk'                                      #
####################################################################

GPT_MODEL = 'gpt2'
GEN_TYPE = 'beam'
NUM_BEAMS = 3
NUM_TOPK = 3
GEN_ARGS = {
    'repetition_penalty': 2.5,
    'length_penalty': 2.5,
    'early_stopping': True,
    'use_cache': True,
    'num_return_sequences': 3,
    'no_repeat_ngram_size': 2
}



# functions, etc.

def prepare_model(gpt_model, fast=True):

    print(col.WARNING + "Preparing model..." + col.ENDC)

    if fast:
        tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_model)

    model = GPT2LMHeadModel.from_pretrained(gpt_model)
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    model.resize_token_embeddings(len(tokenizer))

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    print(col.OKGREEN + "Model prepared!" + col.ENDC)

    return tokenizer, model



def prep_gen(data, for_eval):

    # remove columns
    if for_eval == True:
        data.drop(columns=['annotatorGender', 'annotatorMinority', 'WorkerId', 'HITId',
                        'annotatorPolitics', 'annotatorRace', 'annotatorAge',
                        'dataSource', 'targetCategory', 'sexReason', 'sexPhrase',
                        'targetMinority', 'sexYN', 'whoTarget', 'intentYN',
                        'offensiveYN', 'speakerMinorityYN'], inplace=True)
        # remove nan values
        data = data.dropna(subset = ['targetStereotype'])

    print(col.OKBLUE + "Dataset contains:")
    print(data, col.ENDC)

    # convert to numpy
    data = data.to_numpy()

    if for_eval:
        posts = []
        labels = []
        count = 0

        # posts and labels for eval
        for post in data:
            if count == 0 or post[0] != posts[count - 1]:
                posts.append(post[0])
                labels.append([post[1]])
                count = count + 1
            else:
                new_label = True
                for label in labels[count - 1]:
                    if post[1] == label:
                        new_label = False
                if new_label:
                    labels[count - 1].append(post[1])

        posts = [post + " [SEP] " for post in posts]

        labels = [[" [SEP] ".join(label_group)] if len(label_group) > 1 else label_group for label_group in labels]
        labels = [label for [label] in labels]
        labels = [label.replace(", ", " [SEP] ") for label in labels]
    else:
        posts = [post[0] + " [SEP] " for post in data]

        # [CONCEPTUALISATION] [SEP] [TARGETED GROUP] [RELATION] [IMPLIED STATEMENT]
        labels = [str(post[7]) + " [SEP] " + str(post[4]) + " " + str(post[5]) + " " + str(post[6]) for post in data]
        # [TARGETED GROUP] [RELATION] [IMPLIED STATEMENT] [SEP] [CONCEPTUALISATION]
        # labels = [str(post[4]) + " " + str(post[5]) + " " + str(post[6]) + " [SEP] " + str(post[7]) for post in data]
        # [TARGETED GROUP] [RELATION] [IMPLIED STATEMENT]
        # labels = [str(post[4]) + " " + str(post[5]) + " " + str(post[6]) for post in data]

    return posts, labels



def prep_dataset(tokenizer):

    print(col.WARNING + "Loading datasets..." + col.ENDC)

    # load from file
    data_trn = pd.read_csv(DATA_TRN_PATH)
    data_dev = pd.read_csv(DATA_DEV_PATH)
    data_tst = pd.read_csv(DATA_TST_PATH)

    # display all the columns
    pd.options.display.max_columns = None

    # shuffle only training dataset
    data_trn = data_trn.sample(frac=1).reset_index(drop=True)

    # a subset of data rows to make training quicker
    if SUBSET == True:
        data_trn = data_trn[:16000]
        data_dev = data_dev[:2000]
        data_tst = data_tst[:2000]

    # preprocess data
    posts_trn, labels_trn = prep_gen(data_trn, for_eval=False)
    posts_dev, labels_dev = prep_gen(data_dev, for_eval=True)
    posts_tst, labels_tst = prep_gen(data_tst, for_eval=True)
    labels_trn = [label + tokenizer.eos_token for label in labels_trn]

    # perform encoding
    encodings_trn = tokenizer(posts_trn, padding=False, truncation=True)
    encodings_dev = tokenizer(posts_dev, padding=False, truncation=True)
    encodings_tst = tokenizer(posts_tst, padding=False, truncation=True)
    labels_encodings_trn = tokenizer(labels_trn, padding=False, truncation=True)
    labels_encodings_dev = tokenizer(labels_dev, padding=False, truncation=True)
    labels_encodings_tst = tokenizer(labels_tst, padding=False, truncation=True)

    # convert to Dataset
    dataset_trn = DatasetGen(encodings_trn, labels_encodings_trn, for_eval=False)
    dataset_dev = DatasetGen(encodings_dev, labels_encodings_dev, for_eval=True)
    dataset_tst = DatasetGen(encodings_tst, labels_encodings_tst, for_eval=True)

    print(col.OKGREEN + "Datasets loaded!" + col.ENDC)

    return dataset_trn, dataset_dev, dataset_tst



def prep_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(col.OKBLUE + 'There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0), col.ENDC)
    else:
        print(col.OKBLUE + 'No GPU available, using the CPU instead.' + col.ENDC)
        device = torch.device("cpu")

    return device



def prep_badwords(file, tokenizer):

    f = open(file)
    data = json.load(f)
    badwords_input_ids = tokenizer(data, add_prefix_space=True).input_ids
    f.close()

    return badwords_input_ids
