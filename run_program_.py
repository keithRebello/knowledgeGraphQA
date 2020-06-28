#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
import numpy as np
import random
import os
import pickle
import sys
import argparse
import re
import augment_process_dataset as apd
import pandas as pd
from nltk.corpus import stopwords
from itertools import compress
from evaluation import evaluation, get_span
from argparse import ArgumentParser
from torchtext import data
from sklearn.metrics.pairwise import euclidean_distances
from fuzzywuzzy import fuzz
from util import www2fb, processed_text, clean_uri


# In[24]:


#### THIS IS TO BE RUN ONLY ONCE #####

# In[25]:


class Args:
    seed = 345
    cuda = False
    gpu = -1
    embed_dim = 250
    batch_size = 16
    dete_model = 'dete_best_model.pt'
    entity_model = 'entity_best_model.pt'
    pred_model='pred_best_model.pt'
    output='preprocess'
    data = './data/penn'
    model = 'LSTM'
    emsize = 200
    nhid = 200
args=Args()
args.dete_model = os.path.join(args.output, args.dete_model)
args.entity_model = os.path.join(args.output, args.entity_model)
args.pred_model = os.path.join(args.output, args.pred_model)
whhowset = [{'what', 'how', 'where', 'who', 'which', 'whom'},
            {'in which', 'what is', "what 's", 'what are', 'what was', 'what were', 'where is', 'where are',
             'where was', 'where were', 'who is', 'who was', 'who are', 'how is', 'what did'},
            {'what kind of', 'what kinds of', 'what type of', 'what types of', 'what sort of'}]


# In[26]:


'''def entity_predict(dataset_iter):
    model.eval()
    dataset_iter.init_epoch()
    gold_list = []
    pred_list = []
    dete_result = []
    question_list = []
    for data_batch_idx, data_batch in enumerate(dataset_iter):
        #batch_size = data_batch.text.size()[1]
        answer = torch.max(model(data_batch), 1)[1].view(data_batch.ed.size())
        answer[(data_batch.text.data == 1)] = 1
        answer = np.transpose(answer.cpu().data.numpy())
        gold_list.append(np.transpose(data_batch.ed.cpu().data.numpy()))
        index_question = np.transpose(data_batch.text.cpu().data.numpy())
        question_array = index2word[index_question]
        dete_result.extend(answer)
        question_list.extend(question_array)
        pred_list.append(answer)
    return dete_result, question_list'''

# In[27]:
'''def compute_reach_dic(matched_mid):
    reach_dic = {}  # reach_dic[head_id] = (pred_id, tail_id)
    with open(os.path.join(args.output, 'transE_train.txt'), 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            head_id = items[0]
            if head_id in matched_mid and items[2] in pre_dic:
                if reach_dic.get(head_id) is None:
                    reach_dic[head_id] = [pre_dic[items[2]]]
                else:
                    reach_dic[head_id].append(pre_dic[items[2]])
    return reach_dic'''

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
# In[37]:

#### NOTE : THIS RUNS AS LONG AS REQUIRED #####
# In[38]:
###################### TAKING INPUT AND GENERATING THE OTHER FIELDS FOR QA OUTPUT ########################s
TEXT = data.Field(lower=True)
ED = data.Field()
ip = input('enter a question: ') # Input is taken from the User here
ip = processed_text(ip)

outfile = open(os.path.join(args.output, 'input_file.txt'), 'a')
tok = apd.reverseLinking(ip,None)
tok = tok[1]
outfile.write('{}\t{}\t\n'.format(ip, tok))
outfile.close() # INPUT gets empty ed field padding

out= data.TabularDataset(path=os.path.join(args.output, 'input_file.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
train = data.TabularDataset(path=os.path.join(args.output, 'dete_train.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)]
dev, test = data.TabularDataset.splits(path=args.output, validation='valid.txt', test='test.txt', format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)
ED.build_vocab(train, dev)
total_num = len(test)

if args.gpu == -1: # Load all tensors onto the CPU
    test_iter = data.Iterator(out, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False, 
                              sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage)
    model.config.cuda = False
else:
    test_iter = data.Iterator(out, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                              repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
index2tag = np.array(ED.vocab.itos)
idxO = int(np.where(index2tag == 'O')[0][0])  # Index for 'O'
idxI = int(np.where(index2tag == 'I')[0][0])  # Index for 'I'
index2word = np.array(TEXT.vocab.itos)
model.eval()
dataset_iter.init_epoch()
gold_list = []
pred_list = []
dete_result = []
question_list = []
for data_batch_idx, data_batch in enumerate(dataset_iter):
    #batch_size = data_batch.text.size()[1]
    answer = torch.max(model(data_batch), 1)[1].view(data_batch.ed.size())
    answer[(data_batch.text.data == 1)] = 1
    answer = np.transpose(answer.cpu().data.numpy())
    gold_list.append(np.transpose(data_batch.ed.cpu().data.numpy()))
    index_question = np.transpose(data_batch.text.cpu().data.numpy())
    question_array = index2word[index_question]
    dete_result.extend(answer)
    question_list.extend(question_array)
    pred_list.append(answer)
del model

realed=[]
for i in range(len(dete_result[0])):
    if dete_result[0][i] == 2:
        realed.append('O')
    if dete_result[0][i] == 3:
        realed.append('I')
    realed.append(' ')
del realed[-1]
realed=''.join(realed)



emp = ''
outfile = open(os.path.join(args.output, 'test.txt'), 'a')
outfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(emp,emp,emp,emp,emp,ip, realed))
outfile.close() # I/P is given proper ed field and is padded and added to the test file

#sample question -- m.0h5t1m8	who produced the film woodstock villa


# In[39]:


######################## Entity Detection  ########################
TEXT = data.Field(lower=True)
ED = data.Field()
train = data.TabularDataset(path=os.path.join(args.output, 'dete_train.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', ED)]
out= data.TabularDataset(path=os.path.join(args.output, 'input_file.txt'), format='tsv', fields=[('text', TEXT), ('ed', ED)])
dev, test = data.TabularDataset.splits(path=args.output, validation='valid.txt', test='test.txt', format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)
ED.build_vocab(train, dev)
total_num = len(test)

# load the model
if args.gpu == -1: # Load all tensors onto the CPU
    test_iter = data.Iterator(test, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False, 
                              sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage)
    model.config.cuda = False
else:
    test_iter = data.Iterator(test, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                              repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    model = torch.load(args.dete_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
index2tag = np.array(ED.vocab.itos)
idxO = int(np.where(index2tag == 'O')[0][0])  # Index for 'O'
idxI = int(np.where(index2tag == 'I')[0][0])  # Index for 'I'
index2word = np.array(TEXT.vocab.itos)
# run the model on the test set and write the output to a file
model.eval()
dataset_iter.init_epoch()
gold_list = []
pred_list = []
dete_result = []
question_list = []
for data_batch_idx, data_batch in enumerate(dataset_iter):
    #batch_size = data_batch.text.size()[1]
    answer = torch.max(model(data_batch), 1)[1].view(data_batch.ed.size())
    answer[(data_batch.text.data == 1)] = 1
    answer = np.transpose(answer.cpu().data.numpy())
    gold_list.append(np.transpose(data_batch.ed.cpu().data.numpy()))
    index_question = np.transpose(data_batch.text.cpu().data.numpy())
    question_array = index2word[index_question]
    dete_result.extend(answer)
    question_list.extend(question_array)
    pred_list.append(answer)
del model


# In[40]:


######################## Find matched names  ########################
mid_dic, mid_num_dic = {}, {}  # Dictionary for MID
for line in open(os.path.join(args.output, 'entity2id.txt'), 'r'):
    items = line.strip().split("\t")
    mid_dic[items[0]] = int(items[1])
    mid_num_dic[int(items[1])] = items[0]

pre_dic, pre_num_dic = {}, {}  # Dictionary for predicates
match_pool = []
for line in open(os.path.join(args.output, 'relation2id.txt'), 'r'):
    items = line.strip().split("\t")
    match_pool = match_pool + items[0].replace('.', ' ').replace('_', ' ').split()
    pre_dic[items[0]] = int(items[1])
    pre_num_dic[int(items[1])] = items[0]
#names_map = {}
entities_emb = np.fromfile(os.path.join(args.output, 'entities_emb.bin'), dtype=np.float32).reshape((len(mid_dic), args.embed_dim))
predicates_emb = np.fromfile(os.path.join(args.output, 'predicates_emb.bin'), dtype=np.float32).reshape((-1, args.embed_dim))
index_names = {}
for i, line in enumerate(open(os.path.join(args.output, 'names.trimmed.txt'), 'r')):
    items = line.strip().split("\t")
    entity = items[0]
    literal = items[1].strip()
    if literal != "":
        #if names_map.get(entity) is None or len(names_map[entity].split()) > len(literal.split()):
        #    names_map[entity] = literal
        if index_names.get(literal) is None:
            index_names[literal] = [entity]
        else:
            index_names[literal].append(entity)
for fname in ["train.txt", "valid.txt"]:
    with open(os.path.join(args.output, fname), 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            if items[2] != '<UNK>' and mid_dic.get(items[1]) is not None:
                if index_names.get(items[2]) is None:
                    index_names[items[2]] = [items[1]]
                else:
                    index_names[items[2]].append(items[1])
head_mid_idx = [[] for i in range(total_num)]# [[head1,head2,...], [head1,head2,...], ...]
match_pool = set(match_pool + stopwords.words('english') + ["'s"])
dete_tokens_list, filter_q = [], []

for i, question in enumerate(question_list):
    question = [token for token in question if token != '<pad>']
    pred_span = get_span(dete_result[i], index2tag, type=False)
    tokens_list, dete_tokens, st, en, changed = [], [], 0, 0, 0
    for st, en in pred_span:
        tokens = question[st:en]
        tokens_list.append(tokens)
        if index_names.get(' '.join(tokens)) is not None:  # important
            dete_tokens.append(' '.join(tokens))
            head_mid_idx[i].append(' '.join(tokens))
    if len(question) > 2:
        for j in range(3, 0, -1):
            if ' '.join(question[0:j]) in whhowset[j - 1]:
                changed = j
                del question[0:j]
                continue
    tokens_list.append(question)
    filter_q.append(' '.join(question[:st - changed] + question[en - changed:]))
    if not head_mid_idx[i]:
        dete_tokens = question
        for tokens in tokens_list:
            grams = []
            maxlen = len(tokens)
            for j in range(maxlen - 1, 1, -1):
                for token in [tokens[idx:idx + j] for idx in range(maxlen - j + 1)]:
                    grams.append(' '.join(token))
            for gram in grams:
                if index_names.get(gram) is not None:
                    head_mid_idx[i].append(gram)
                    break
            for j, token in enumerate(tokens):
                if token not in match_pool:
                    tokens = tokens[j:]
                    break
            if index_names.get(' '.join(tokens)) is not None:
                head_mid_idx[i].append(' '.join(tokens))
            tokens = tokens[::-1]
            for j, token in enumerate(tokens):
                if token not in match_pool:
                    tokens = tokens[j:]
                    break
            tokens = tokens[::-1]
            if index_names.get(' '.join(tokens)) is not None:
                head_mid_idx[i].append(' '.join(tokens))
    dete_tokens_list.append(' '.join(dete_tokens))
id_match = set()
match_mid_list = []
tupleset = []
for i, names in enumerate(head_mid_idx):
    tuplelist = []
    for name in names:
        mids = index_names[name]
        match_mid_list.extend(mids)
        for mid in mids:
            if mid_dic.get(mid) is not None:
                tuplelist.append((mid, name))
    tupleset.extend(tuplelist)
    head_mid_idx[i] = list(set(tuplelist))
    if tuplelist:
        id_match.add(i)
tupleset = set(tupleset)
with open ('tuple_topic.txt', 'rb') as fp:
    tuple_topic = pickle.load(fp)


# In[41]:


######################## Learn entity representation  ########################
head_emb = np.zeros((total_num, args.embed_dim))
TEXT = data.Field(lower=True)
ED = data.Field(sequential=False, use_vocab=False)
train, dev = data.TabularDataset.splits(path=args.output, train='entity_train.txt', validation='entity_valid.txt', format='tsv', fields=[('text', TEXT), ('mid', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)]
test = data.TabularDataset(path=os.path.join(args.output,'test.txt'), format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)  # training data includes validation data
# load the model
if args.gpu == -1:  # Load all tensors onto the CPU
    test_iter = data.Iterator(test, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False, 
                              sort_within_batch=False)
    er_model = torch.load(args.entity_model, map_location=lambda storage, loc: storage)
    er_model.config.cuda = False
else:
    test_iter = data.Iterator(test, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                              repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    er_model = torch.load(args.entity_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
er_model.eval()
test_iter.init_epoch()
baseidx = 0
for data_batch_idx, data_batch in enumerate(test_iter):
    batch_size = data_batch.text.size()[1]
    scores = er_model(data_batch).cpu().data.numpy()
    for i in range(batch_size):
        head_emb[baseidx + i] = scores[i]
    baseidx = baseidx + batch_size
del er_model


# In[42]:


######################## Learn predicate representation  ########################
TEXT = data.Field(lower=True)
ED = data.Field(sequential=False, use_vocab=False)
train, dev = data.TabularDataset.splits(path=args.output, train='pred_train.txt', validation='pred_valid.txt', format='tsv', fields=[('text', TEXT), ('mid', ED)])
field = [('id', None), ('sub', None), ('entity', None), ('relation', None), ('obj', None), ('text', TEXT), ('ed', None)]
test = data.TabularDataset(path=os.path.join(args.output,'test.txt'), format='tsv', fields=field)
TEXT.build_vocab(train, dev, test)
# load the model
if args.gpu == -1:  # Load all tensors onto the CPU
    test_iter = data.Iterator(test, batch_size=args.batch_size, train=False, repeat=False, sort=False, shuffle=False, 
                              sort_within_batch=False)
    pr_model = torch.load(args.pred_model, map_location=lambda storage, loc: storage)
    pr_model.config.cuda = False
else:
    test_iter = data.Iterator(test, batch_size=args.batch_size, device=torch.device('cuda', args.gpu), train=False,
                              repeat=False, sort=False, shuffle=False, sort_within_batch=False)
    pr_model = torch.load(args.pred_model, map_location=lambda storage, loc: storage.cuda(args.gpu))
pr_model.eval()
test_iter.init_epoch()
baseidx = 0
pred_emb = np.zeros((total_num, args.embed_dim))
for data_batch_idx, data_batch in enumerate(test_iter):
    batch_size = data_batch.text.size()[1]
    scores = pr_model(data_batch).cpu().data.numpy()
    for i in range(batch_size):
        s = scores[i]
        pred_emb[baseidx + i] = s        
    baseidx = baseidx + batch_size
del pr_model


# In[43]:


############################ CALCULATION OF THE DISTANCE BETWEEN THE PREDICATES AND ENTITIES #######################
notmatch = list(set(range(0, total_num)).symmetric_difference(id_match))
notmatch_idx = euclidean_distances(head_emb[notmatch], entities_emb, squared=True).argsort(axis=1)
for idx, i in enumerate(notmatch):
    for j in notmatch_idx[idx, 0:40]:
        mid = mid_num_dic[j]
        head_mid_idx[i].append((mid, None))
        match_mid_list.append(mid)
        correct, mid_num = 0, 0
for i, head_ids in enumerate(head_mid_idx):
    mids = set()
    for (head_id, name) in head_ids:
        mids.add(head_id)
    mid_num += len(mids)# reach_dic[head_id] = (pred_id, tail_id)
reach_dic = set(match_mid_list)
with open(os.path.join(args.output, 'transE_train.txt'), 'r') as f:
    for line in f:
        items = line.strip().split("\t")
        head_id = items[0]
        if head_id in matched_mid and items[2] in pre_dic:
            if reach_dic.get(head_id) is None:
                reach_dic[head_id] = [pre_dic[items[2]]]
            else:
                reach_dic[head_id].append(pre_dic[items[2]])
learned_pred, learned_fact, learned_head = [-1] * total_num, {}, [-1] * total_num
alpha1, alpha3 = .39, .43
for i, head_ids in enumerate(head_mid_idx[-1]):  # head_ids is mids
    i = total_num - 1
    answers = []
    head_id = head_ids[0]
    name = head_ids[1]
    mid_score = np.sqrt(np.sum(np.power(entities_emb[mid_dic[head_id]] - head_emb[i], 2)))
    name_score = - .003 * fuzz.ratio(name, dete_tokens_list[i])
    if (head_id, name) in tuple_topic:
        name_score -= .18
    if reach_dic.get(head_id) is not None:
        for pred_id in reach_dic[head_id]:  # reach_dic[head_id] = pred_id are numbers
                rel_names = - .017 * fuzz.ratio(pre_num_dic[pred_id].replace('.', ' ').replace('_', ' '), filter_q[i]) #0.017
                rel_score = np.sqrt(np.sum(np.power(predicates_emb[pred_id] - pred_emb[i], 2))) + rel_names
                tai_score = np.sqrt(np.sum(
                    np.power(predicates_emb[pred_id] + entities_emb[mid_dic[head_id]] - head_emb[i] - pred_emb[i], 2)))
                answers.append((head_id, pred_id, alpha1 * mid_score + rel_score + alpha3 * tai_score + name_score))
    if answers:
        answers.sort(key=lambda x: x[2])
        learned_head[i] = answers[0][0]
        learned_pred[i] = answers[0][1]
        learned_fact[' '.join([learned_head[i], pre_num_dic[learned_pred[i]]])] = i
learned_tail = [[] for i in range(total_num)]
for line in open(os.path.join(args.output, 'cleanedFB.txt'), 'r'):
    items = line.strip().split("\t")
    if learned_fact.get(' '.join([items[0], items[2]])) is not None:
        learned_tail[learned_fact[' '.join([items[0], items[2]])]].extend(items[1].split())


# In[44]:


#################### OUTPUT GENERATION ###############################
matches = []
is_empty = 0
if len(learned_tail[-1]) == 0:
    matches.append(" Sorry bro, idk")
    is_empty = 1
else:
    stringToMatch = learned_tail[-1][0]
    matchedLine = ''
    with open('preprocess/heads_toes.txt', 'r') as file:
        for line in file:
            potmatch = line.strip().split('\t')
            if stringToMatch == potmatch[0]:
                #matchedLine = line.strip().split('\t')
                matches.append(potmatch[1])
                is_empty = 1
                break
            if is_empty == 1:
                break
readFile = open("preprocess/test.txt")
lines = readFile.readlines()
readFile.close()
w = open("preprocess/test.txt",'w')
w.writelines([item for item in lines[:-1]])
w.close()
if is_empty == 0:
    print('I dont know bro')
else:
    print(matches) # PRINTS THE OUTPUT