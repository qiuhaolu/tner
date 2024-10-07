from transformers import pipeline
import pandas as pd
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
import torch
from tqdm import tqdm
from utils import *

# this code is mostly used for evaluation
# gpt4, medtuned



def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["Token"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["Tag"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def parse_medtuned_outputs(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
    
    documents = content.split('######')
    result = []

    for doc in documents:
        lines = doc.strip().split('\n')
        bio_tags = []
        for line in lines:
            if line:  # Ensure the line is not empty
                parts = line.replace('[','').replace(']','').split(': ')
                if len(parts) == 2:
                    # Append only the tag
                    bio_tags.append(parts[1].strip())
                else:
                    # Assign 'O' as the tag if only token is present
                    bio_tags.append('O')
        result.append(bio_tags)

    return result

### the current gpt4 output is a list of labels, without the input tokens like medtuned output
def parse_gpt_outputs(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
    
    documents = content.split('######')
    result = []

    for doc in documents:
        lines = doc.strip().split('\n')
        bio_tags = []
        for line in lines:
            if line:  # Ensure the line is not empty
                bio_tags.append(line.strip())
        result.append(bio_tags)

    return result




def modify_result_list_rare(result):
    modified_result = []
    for sublist in result:
        modified_sublist = []
        for tag in sublist:
            if 'RAREDISEASE' in tag or 'SKINRAREDISEASE' in tag:
                modified_sublist.append(tag[0])  # Keep only 'B' or 'I'
            else:
                modified_sublist.append(tag)
        modified_result.append(modified_sublist)
    return modified_result

def modify_result_list_disease(result):
    modified_result = []
    for sublist in result:
        modified_sublist = []
        for tag in sublist:
            if '-DISEASE' in tag:
                modified_sublist.append(tag[0])  # Keep only 'B' or 'I'
            else:
                modified_sublist.append(tag)
        modified_result.append(modified_sublist)
    return modified_result


def align_preds(result, truelabels, mode):
    assert len(result) == len(truelabels)
    assert mode in ['DISEASE','RAREDISEASE','SKINRAREDISEASE','SIGN','SYMPTOM','ANAPHOR','ALL']
    modified_result = []
    #print(result[0:10])
    #exit()
    for i in range(len(result)):
        len_result_i = len(result[i])
        len_truelabels_i = len(truelabels[i])

        # Initialize the aligned sublist
        aligned_sublist = []            
        for j in range(len_truelabels_i):
            if j < len_result_i and mode != 'ALL':
                tag = result[i][j]
                #print(tag)
                # Replace B with B-DISEASE and I with I-DISEASE
                if 'B' in tag: ## this is for GPT4
                #if tag == 'B':
                    aligned_sublist.append('B-'+mode)
                elif 'I' in tag:
                #elif tag == 'I':
                    aligned_sublist.append('I-'+mode)
                else:
                    aligned_sublist.append('O')
            elif j < len_result_i and mode == 'ALL':
                tag = result[i][j]
                aligned_sublist.append(tag)
            else:
                # Pad with 'O' if result sublist is shorter than truelabels sublist
                aligned_sublist.append('O')

        modified_result.append(aligned_sublist)

    return modified_result




train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'

'''
df_train = pd.read_csv(train_path)
print('size of the training dataset: {}'.format(len(df_train)))
df_dev = pd.read_csv(dev_path)
print('size of the development dataset: {}'.format(len(df_dev)))
df_test = pd.read_csv(test_path)
print('size of the test dataset: {}'.format(len(df_test)))
print('datasets loaded!\n')

#number of labels (IOB tags)
tags = df_train['Tag'].unique()
num_tags = df_train['Tag'].nunique()
print('Labels: {}'.format(tags))
print('Nr of labels: {}'.format(num_tags))


# Overall statistics for the number of words in each text
count_df_train = df_train.groupby('Sentence_Num_Global').count()
#print(count_df_train)
statistics_train = count_df_train['Token'].describe()
print('\nSome statistics of the sentences in the training dataset:')
print(statistics_train)

count_df_dev = df_dev.groupby('Sentence_Num_Global').count()
statistics_dev = count_df_dev['Token'].describe()
print('\nSome statistics of the sentences in the development dataset:')
print(statistics_dev)

#The lenth of the longest sentence. Lenght is the number of words.
MAX_LEN_TRAIN = int(statistics_train['max'])
MAX_LEN_DEV = int(statistics_dev['max'])
MAX_LEN = max(MAX_LEN_TRAIN, MAX_LEN_DEV)
print('\n')
print('The maximum length of sentences in TRAIN is: ', MAX_LEN_TRAIN)
print('The maximum length of sentences in DEV is: ', MAX_LEN_DEV)
print('The maximum length of sentences in TOTAL is:', MAX_LEN)

'''


# Load your CSV file
dev_data = pd.read_csv(dev_path)
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

raredis_datasets = create_raredis_datasets(train_path, dev_path, test_path)



label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)



metric = evaluate.load("seqeval")
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}



true_labels = raredis_datasets['test']['Tag']
true_labels_str = [[id2label[label] for label in label_seq if label != -100] for label_seq in true_labels]

#print(true_labels[0])
#exit()
#{0: 'B-ANAPHOR', 1: 'B-DISEASE', 2: 'B-RAREDISEASE', 3: 'B-SIGN', 4: 'B-SKINRAREDISEASE', 5: 'B-SYMPTOM', 6: 'I-ANAPHOR', 7: 'I-DISEASE', 
#8: 'I-RAREDISEASE', 9: 'I-SIGN', 10: 'I-SKINRAREDISEASE', 11: 'I-SYMPTOM', 12: 'O'}

eval_path_rare = 'gpt4-1106-preview-chat_[all]_5shot_result_test' 

preds_rare_gpt4 = parse_medtuned_outputs(eval_path_rare)[1:]  
preds_rare_gpt4_aligned = align_preds(preds_rare_gpt4, true_labels_str, 'ALL')

report5  = classification_report(true_labels_str, preds_rare_gpt4_aligned, digits=4)


print(report5)
