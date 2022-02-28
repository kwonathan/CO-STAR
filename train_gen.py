##### import necessary files #####
from colours import col
from utils import *

##### import necessary modules #####
import csv
import torch
import torch.nn.functional as F
import gc
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AdamW

##### cleanup #####
print(col.WARNING + "Cleaning up..." + col.ENDC)
torch.cuda.empty_cache()
gc.collect()
print(col.OKGREEN + "Cleanup done!" + col.ENDC)

##### prepare model #####
tokenizer, model = prepare_model(GPT_MODEL, fast=False)

##### process data #####
dataset_trn, dataset_dev, dataset_tst = prep_dataset(tokenizer)

##### prepare device #####
print(col.WARNING + "Preparing model..." + col.ENDC)
device = prep_device()
model.to(device)
print(col.OKGREEN + "Model prepared!" + col.ENDC)

##### bad words #####
badwords_input_ids = prep_badwords(BADWORDS_PATH, tokenizer)

##### begin training #####
print(col.WARNING + "Training model..." + col.ENDC)

if GEN_TYPE != 'beam' and GEN_TYPE != 'topk':
    print(col.FAIL + "Invalid GEN_TYPE provided. Exiting." + col.ENDC)
    exit()

loader_trn = DataLoader(dataset_trn, batch_size=BATCH_SIZE, shuffle=True)
if TESTING:
    loader_dev = DataLoader(dataset_tst, batch_size=BATCH_SIZE, shuffle=False)
else:
    loader_dev = DataLoader(dataset_dev, batch_size=BATCH_SIZE, shuffle=False)
optim = AdamW(model.parameters(), lr=LEARNING_RATE)
model.parallelize()

# progress bar
num_training_steps = TRAIN_EPOCH * len(loader_trn)
progress_bar = tqdm(range(num_training_steps))

# start training
for epoch in range(TRAIN_EPOCH):

    model.train()

    for batch in loader_trn:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        loss = outputs.loss

        loss.backward()
        optim.step()
        optim.zero_grad()
        progress_bar.update(1)

    ##### evaluating model #####
    print(col.WARNING + "Evaluating model..." + col.ENDC)
    print(col.OKBLUE + "Epoch", epoch, col.ENDC)

    model.eval()

    with torch.no_grad():

        for batch in loader_dev:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            model.config.max_length = len(batch['input_ids'].squeeze().numpy()) + 50

            if GEN_TYPE == 'beam':
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=NUM_BEAMS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=badwords_input_ids,
                    **GEN_ARGS
                )
            elif GEN_TYPE == 'topk':
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_k=NUM_TOPK,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bad_words_ids=badwords_input_ids,
                    **GEN_ARGS
                )

            decoded_inputs = tokenizer.batch_decode(input_ids)
            [decoded_inputs] = decoded_inputs
            decoded_outputs = tokenizer.batch_decode(outputs)
            decoded_outputs = [decoded_output[len(decoded_inputs):] for decoded_output in decoded_outputs]
            decoded_outputs = [decoded_output.replace(tokenizer.eos_token, "") for decoded_output in decoded_outputs]
            decoded_labels = tokenizer.batch_decode(labels)
            [decoded_labels] = decoded_labels
            decoded_labels = decoded_labels.split(" [SEP] ")

            pred_labels = decoded_outputs
            true_labels = decoded_labels

            pred_labels = [label.replace(tokenizer.eos_token, "") for label in pred_labels]
            print(col.OKBLUE + "Post: " + decoded_inputs + col.ENDC)
            print("True:", true_labels)
            print("Pred:", pred_labels)

        print(col.WARNING + 'Saving model...' + col.ENDC)
        model.save_pretrained(MODEL_SAVE_PATH + '.epoch-' + str(epoch))
        tokenizer.save_pretrained(MODEL_SAVE_PATH + '.epoch-' + str(epoch))
        print(col.OKGREEN + 'Model saved!' + col.ENDC)

print(col.OKGREEN + "Finished!" + col.ENDC)
