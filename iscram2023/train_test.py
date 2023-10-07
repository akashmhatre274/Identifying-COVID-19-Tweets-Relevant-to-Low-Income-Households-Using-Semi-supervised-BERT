import os
import copy

#import pickle
import numpy as np
import pandas as pd



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
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import os





model_map= {
    #"tinybert": "huawei-noah/TinyBERT_General_4L_312D",
    #"covidbert": "digitalepidemiologylab/covid-twitter-bert-v2",
    #"distilbert": "distilbert-base-uncased",
    #"bertweet": "vinai/bertweet-base",
    "bertweetcovid": "vinai/bertweet-covid19-base-uncased"
}




def get_device():
    device = ""
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    # If there's a GPU available...
    elif torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device

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




def train_paper(train, val, test, batch_size, total_epoch, labels_in_dst, learning_rate, model_path):
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
        for input_ids, attention_mask, labels in trainloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            output1 = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(output1.logits, labels)
            
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


def test_ST(checkpoint_path, test, labels_in_dst, batch_size):
    global model_map
    test_f1_average = []
    test_precision = []
    test_recall = []
    test_acc = []
    test_f1 = []
    
    device = get_device()  
    
    model = AutoModelForSequenceClassification.from_pretrained(model_map['bertweetcovid'], 
                                                          num_labels=labels_in_dst,
                                                          return_dict=True)
    # model.load_state_dict(torch.load(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load the checkpoint
#     checkpoint = torch.load('checkpoint.pth')

#     # Load the model parameters and optimizer state
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']

    model = model.to(device)
    model.eval()
    flat_prob = []
    pred_labels = []
    
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
            test_acc.append(acc)
            test_f1_average.append(f1_macro)
            test_f1.append(f1_score)
            test_precision.append(precision)
            test_recall.append(recall)
            print("test performance: ", acc, precision, recall,f1_macro, f1_score)
            
    return pred_labels, flat_prob


def test_ST2(checkpoint_path, test, labels_in_dst, batch_size):
    global model_map
    test_f1_average = []
    test_precision = []
    test_recall = []
    test_acc = []
    test_f1 = []
    
    device = get_device()  
    
    model = AutoModelForSequenceClassification.from_pretrained(model_map['bertweetcovid'], 
                                                          num_labels=labels_in_dst,
                                                          return_dict=True)
    model.load_state_dict(torch.load(checkpoint_path))


    model = model.to(device)
    model.eval()
    flat_prob = []
    pred_labels = []
    
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
            test_acc.append(acc)
            test_f1_average.append(f1_macro)
            test_f1.append(f1_score)
            test_precision.append(precision)
            test_recall.append(recall)
            print("test performance: ", acc, precision, recall,f1_macro, f1_score)
            
    return pred_labels, flat_prob




# Generate confusion matrix
def confusion_matrix_plot(y_true, y_pred,
                          classes='',
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          image=None,
                          verbose=0,
                          magnify=1.2,
                          dpi=300):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    y_true needs to contain all possible labels.
    cmap=plt.cm.Blues   # default
    cmap=plt.cm.BuPu
    cmap=plt.cm.GnBu
    cmap=plt.cm.Greens
    cmap=plt.cm.OrRd
    """

    # Class labels setup. If none, generate from y_true y_pred
    classes = list(classes)
    if classes:
        assert len(set(y_true)) == len(classes)
    else:
        classes = set(y_true)
    # Title setup
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize setup
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Print if verbose
    if verbose > 0:
        print(cm)
    # Plot setup
    fig, ax = plt.subplots(facecolor='w')
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * magnify)
    if image:
        fig.savefig(image, dpi=dpi, facecolor='w')
    return cm





