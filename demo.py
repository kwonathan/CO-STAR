import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CS_MODEL_PATH = ''
SBF_MODEL_PATH = ''
GPT_MODEL = 'gpt2'
BADWORDS_PATH = 'badwords.json'

tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL)
tokenizer.add_special_tokens({'sep_token': '[SEP]'})
tokenizer.pad_token = tokenizer.eos_token

f = open(BADWORDS_PATH)
data = json.load(f)
badwords_input_ids = tokenizer(data, add_prefix_space=True).input_ids
f.close()

# settings for beam search decoding; change for top-k
GEN_ARGS = {
    'repetition_penalty': 2.5,
    'length_penalty': 2.5,
    'early_stopping': True,
    'use_cache': True,
    'num_return_sequences': 3,
    'no_repeat_ngram_size': 2,
    'num_beams': 3,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
    'bad_words_ids': badwords_input_ids
}

cs_model = GPT2LMHeadModel.from_pretrained(CS_MODEL_PATH)
sbf_model = GPT2LMHeadModel.from_pretrained(SBF_MODEL_PATH)

cs_model.eval()
sbf_model.eval()

print('==================================================')
print(' DEMO FOR MODELS TRAINED ON THE CO-STAR FRAMEWORK')
print('==================================================')

with torch.no_grad():
    while True:
        post = input('Please enter a post: ')
        post = post.lower().strip()
        post += ' [SEP] '
        encoded_post = tokenizer([post], padding=False, truncation=True)
        input_ids = torch.tensor(encoded_post['input_ids'])
        attention_mask = torch.tensor(encoded_post['attention_mask'])

        cs_model.config.max_length = len(input_ids.squeeze().numpy()) + 50
        sbf_model.config.max_length = len(input_ids.squeeze().numpy()) + 50

        cs_outputs = cs_model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_ARGS)
        sbf_outputs = sbf_model.generate(input_ids=input_ids, attention_mask=attention_mask, **GEN_ARGS)

        cs_outputs = tokenizer.batch_decode(cs_outputs)
        sbf_outputs = tokenizer.batch_decode(sbf_outputs)

        cs_outputs = [output[len(post) - 1:].replace(tokenizer.eos_token, "") for output in cs_outputs]
        sbf_outputs = [output[len(post) - 1:].replace(tokenizer.eos_token, "") for output in sbf_outputs]

        cs_outputs = [output.split(' [SEP] ') for output in cs_outputs]
        cs_outputs = [output for output in cs_outputs if len(output) == 2]

        stereotypes = [output[0] for output in cs_outputs]
        concepts = [output[1] for output in cs_outputs]

        stereotypes = [s.lower().strip() for s in stereotypes]
        concepts = [c.lower().strip() for c in concepts]

        sbf_stereotypes = [s.lower().strip() for s in sbf_outputs]

        stereotypes = [s.rstrip('.') for s in stereotypes]
        concepts = [c.rstrip('.') for c in concepts]

        sbf_stereotypes = [s.rstrip('.') for s in sbf_stereotypes]

        stereotypes = list(set(stereotypes))
        concepts = list(set(concepts))

        sbf_stereotypes = list(set(sbf_stereotypes))

        print('')
        print('-------------------  CO-STAR  --------------------')
        print('Stereotypes:')
        print(*stereotypes, sep='\n')
        print('')
        print('Concepts:')
        print(*concepts, sep='\n')
        print('')
        print('--------------  Social Bias Frames  --------------')
        print('Stereotypes:')
        print(*sbf_stereotypes, sep='\n')
        print('')
