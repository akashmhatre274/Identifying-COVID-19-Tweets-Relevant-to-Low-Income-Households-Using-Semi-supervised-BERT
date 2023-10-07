import numpy as np
import pandas as pd
import pickle


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import matplotlib.pyplot as plt
import gc

# Unique to this file
from transformers import AutoTokenizer, AutoModel
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

def get_device():
    device = ""
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

model_map= {
    #"tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    #"covidbert": "digitalepidemiologylab/covid-twitter-bert-v2",
    #"distilbert": "distilbert-base-uncased",
    #"bertweet": "vinai/bertweet-base",
    "bertweetcovid": "vinai/bertweet-covid19-base-uncased"
}


# Depending on your GPU you can either increase or decrease this value
batch_size = 16
total_epoch = 10
learning_rate = 1e-5


# Find out how many labels are in the dataset
with open('/workspace/workspace/iscram2023/covid_dataset/5_class_map.pkl','rb') as f:
    labels = pickle.load(f)
labels_in_dst = len(labels)

def compute_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
#     rounded_preds = torch.nn.functional.softmax(preds)
    softmax = torch.nn.Softmax(dim=1)
    rounded_preds = softmax(preds)
    _, indices = torch.max(rounded_preds, 1)
    rounded_preds = indices
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    
    y_pred = np.array(rounded_preds.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    result = precision_recall_fscore_support(y_true, y_pred, average='macro')
    #f1_average = (result[2][0]+result[2][2])/2
    f1 = f1_score(y_true,y_pred,average='weighted')
    # print(classification_report(y_true, y_pred, digits=4))
    return acc, result[0], result[1], result[2], f1

# This allows sample weight for each training example, when calculate loss 
def train_teacher(train, val,test, batch_size, total_epoch, labels_in_dst, learning_rate, self_train, model_path):
    global model_map

    trainloader = DataLoader(train, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(val, shuffle=False, batch_size=batch_size)
    
    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(model_map['bertweetcovid'], 
                                                          num_labels=labels_in_dst,
                                                          return_dict=True)
    model = model.to(device)
    gc.collect()

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    sum_loss = []
    sum_val = []
    
    val_f1_average = []

    for epoch in range(0, total_epoch):
        print('Epoch:', epoch)
        train_loss, valid_loss = [], []
        
        model.train()
        
        step = 0
        total_train_accuracy = 0
        
        for input_ids, attention_mask, labels, sample_weight in trainloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            sample_weight = sample_weight.to(device)

            
            optimizer.zero_grad()
            output1 = model(input_ids, attention_mask=attention_mask)
            if not self_train:
                loss = F.cross_entropy(output1.logits, labels)
            else:
                logits = output1.logits
                # redefine loss that including the sample weights
                # loss = BCEWithLogitsLoss(logits, b_labels, weight=b_sample_weights)
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, labels_in_dst), lables.view(-1))
                loss = torch.mean(loss * sample_weights) # element wise multiplication
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        sum_loss.append(sum(train_loss)/len(train))  
        print('Loss: {:.4f}'.format(sum_loss[epoch-1]))

#       evaluation part 
        model.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for input_ids, attention_mask, labels in valloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                output = model(input_ids, attention_mask=attention_mask)
                predictions.append(output.logits.clone().detach())
                true_labels.append(labels.clone().detach())
            predictions = torch.cat(predictions)
            true_labels = torch.cat(true_labels)
            predictions = predictions.cpu()
            true_labels = true_labels.cpu()

            # val_f1 is weighted f1 
            acc, precision, recall, f1_macro, val_f1  = compute_accuracy(predictions, true_labels)
            print("validation performance: ", acc, precision, recall, f1_macro, val_f1)
            
            
            best_f1 = max(val_f1_average, default=-1)
            best_model_state = ''
            # Save the best model seen so far
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_f1': best_f1
                            }, model_path)
            
            val_f1_average.append(val_f1)
    
        # test
        model.eval()
        testloader = DataLoader(test, shuffle=False, batch_size=batch_size)

        with torch.no_grad():
            predictions = []
            true_labels = []
            pred_prob = []
            for input_ids, attention_mask, labels in testloader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits.clone().detach()

                predictions.append(logits)
                true_labels.append(labels.clone().detach())

                softmax = torch.nn.Softmax(dim=1)
                prob_batch = softmax(logits)
                prob_batch = prob_batch.cpu().numpy()
                pred_prob.append(prob_batch)

            predictions = torch.cat(predictions)
            true_labels = torch.cat(true_labels)
            predictions = predictions.cpu()
            true_labels = true_labels.cpu()

            flat_prob = np.concatenate(pred_prob, axis=0)

            pred_labels = np.argmax(flat_prob, axis=1).flatten()

            acc, precision, recall,f1_macro, f1_score  = compute_accuracy(predictions, true_labels)

            print("test model performance after all epochs: ", acc, precision, recall,f1_macro, f1_score)

df_bertweetcovid = pd.read_csv('/workspace/workspace/iscram2023/covid_dataset/splits/19k_bertweetcovid_probs-2.csv')
df_bertweet = pd.read_csv('/workspace/workspace/iscram2023/covid_dataset/splits/19k_bertweet_probs-2.csv')
df_covidbert = pd.read_csv('/workspace/workspace/iscram2023/covid_dataset/splits/19k_covidbert_probs-2.csv')

df_res = df_bertweetcovid
df_covidbert_cols = ['covidbert_class0', 'covidbert_class1', 'covidbert_class2', 'covidbert_class3', 'covidbert_class4']
df_bertweet_cols = ['bertweet_class0', 'bertweet_class1', 'bertweet_class2', 'bertweet_class3', 'bertweet_class4']

df_res = df_res.join(df_covidbert[df_covidbert_cols])
df_res = df_res.join(df_bertweet[df_bertweet_cols])

df_res['class0'] = 0
df_res['class1'] = 0
df_res['class2'] = 0
df_res['class3'] = 0
df_res['class4'] = 0

# df_res.to_csv('19k_combined.csv')

for i in range(df_res.shape[0]):
    df_res.iloc[i, df_res.columns.get_loc('class0')] = ((df_res.iloc[i, df_res.columns.get_loc('bertweetcovid_class0')] + df_res.iloc[i, df_res.columns.get_loc('covidbert_class0')] + df_res.iloc[i, df_res.columns.get_loc('bertweet_class0')]) / 3)
    df_res.iloc[i, df_res.columns.get_loc('class1')] = ((df_res.iloc[i, df_res.columns.get_loc('bertweetcovid_class1')] + df_res.iloc[i, df_res.columns.get_loc('covidbert_class1')] + df_res.iloc[i, df_res.columns.get_loc('bertweet_class1')]) / 3)
    df_res.iloc[i, df_res.columns.get_loc('class2')] = ((df_res.iloc[i, df_res.columns.get_loc('bertweetcovid_class2')] + df_res.iloc[i, df_res.columns.get_loc('covidbert_class2')] + df_res.iloc[i, df_res.columns.get_loc('bertweet_class2')]) / 3)
    df_res.iloc[i, df_res.columns.get_loc('class3')] = ((df_res.iloc[i, df_res.columns.get_loc('bertweetcovid_class3')] + df_res.iloc[i, df_res.columns.get_loc('covidbert_class3')] + df_res.iloc[i, df_res.columns.get_loc('bertweet_class3')]) / 3)
    df_res.iloc[i, df_res.columns.get_loc('class4')] = ((df_res.iloc[i, df_res.columns.get_loc('bertweetcovid_class4')] + df_res.iloc[i, df_res.columns.get_loc('covidbert_class4')] + df_res.iloc[i, df_res.columns.get_loc('bertweet_class4')]) / 3)

print(df_res)
print(i)
print((df_res['bertweetcovid_class0'][i] + df_res['covidbert_class0'][i] + df_res['bertweet_class0'][i]) / 3)

df_res.to_csv('/workspace/workspace/iscram2023/covid_dataset/splits/19k_avg_v2.1.csv')

#df_19k = pd.read_csv('drive/My Drive/iscram2023/covid_dataset/splits/19k_bertweetcovid_probs.csv')
df_19k_cols = ["Tweet Text", 'Label', 'clean_tweet', 'class0','class1', 'class2', 'class3', 'class4']
pred_probs = df_res[['class0', 'class1', 'class2', 'class3', 'class4']].to_numpy()
df_19k = df_res[df_19k_cols]
#pred_probs

# use pandas to get hard labeled unlabeled data
# but maybe can select directly from TensorDataset, then no need to reclean and retokenize selected tweets

class0_top100 = pred_probs[:, 0].flatten().argsort()[-500:]
print(len(class0_top100))
df_19k.loc[class0_top100, 'Label'] = 0
df_unlabeled_selected = df_19k.iloc[class0_top100]
top500_index = class0_top100.copy()


for i in range(1, labels_in_dst):
    classi_top100 = pred_probs[:, i].flatten().argsort()[-500:]
    df_19k.loc[classi_top100, 'Label'] = i
    
    top500_index = np.concatenate((top500_index, classi_top100))
    df_unlabeled_selected = pd.concat([df_unlabeled_selected, df_19k.iloc[classi_top100]])
print(len(top500_index))
print(len(df_19k))
df_19k.drop(top500_index, inplace=True)
df_19k = df_19k.reset_index(drop=True)
print(len(df_19k))  

pred_probs_top500 = pred_probs[top500_index]
print(pred_probs_top500.shape)

models = {
#     "tinybert": ("huawei-noah/TinyBERT_General_4L_312D",512),
     "covidbert": ("digitalepidemiologylab/covid-twitter-bert-v2",512),
#     "distilbert": ("distilbert-base-uncased",512),
     "bertweet": ("vinai/bertweet-base",130),
    "bertweetcovid": ("vinai/bertweet-covid19-base-uncased",130)
}

data_x = df_unlabeled_selected['clean_tweet'].values.tolist()
data_y = df_unlabeled_selected['Label'].values.tolist()

df_train = pd.read_csv('/workspace/workspace/iscram2023/covid_dataset/splits/train.csv_cleaned.csv')
x_train = df_train['clean_tweet'].values.tolist()
y_train = df_train['Label'].values.tolist()

print(len(data_x))
print(len(x_train), len(y_train))

new_train_x = x_train + data_x
new_train_y = y_train + data_y
print(len(new_train_x), len(new_train_y))


for model_type,(model_name,sequence_length) in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False, normalization = True)

    tokenized_x = tokenizer(new_train_x, return_tensors='pt', padding=True, truncation=True, max_length=sequence_length)
    x_input_ids = tokenized_x['input_ids']
    x_attention_mask = tokenized_x['attention_mask']
    labels = torch.tensor(new_train_y, dtype=torch.long)
    
    # weight is just one for each selected hard labeled unlabeled tweet
    sample_weights = torch.ones(len(new_train_y)).type(torch.FloatTensor)
    
    new_train_dataset = TensorDataset(x_input_ids, x_attention_mask, labels, sample_weights )

import os
cwd = os.getcwd()

dst_path = '/workspace/workspace/iscram2023/preprocessed_data/{}.dst'
dst_path = os.path.join(cwd,'/workspace/workspace/iscram2023/preprocessed_data/{}-{{}}.dst'.format("bertweetcovid"))

train = torch.load(dst_path.format('train'))
val = torch.load(dst_path.format('val'))
test = torch.load(dst_path.format('test'))
unlabeled = torch.load(dst_path.format('19k'))

batch_size = 16
total_epoch = 10
train_teacher(new_train_dataset, val, test, batch_size, total_epoch, labels_in_dst, learning_rate, self_train=False, model_path="bertweetcovid_ST_1iter_500eachclass.pth")
for i in range(5):
    train_teacher(new_train_dataset, val, test, batch_size, total_epoch, labels_in_dst, learning_rate, self_train=False, model_path=str(i)+"bertweetcovid_ST_1iter_500each16batch10epochs.pth")
