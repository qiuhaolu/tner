from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import pandas as pd
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report



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
    #print(examples['Token'])
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




train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'


dev_data = pd.read_csv(dev_path)
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

dev_grouped_data = dev_data.groupby("Sentence_Num_Global").agg({'Token': lambda x: list(x), 'Tag': lambda x: list(x)})
dev_grouped_data.reset_index(inplace=True)

test_grouped_data = test_data.groupby("Sentence_Num_Global").agg({'Token': lambda x: list(x), 'Tag': lambda x: list(x)})
test_grouped_data.reset_index(inplace=True)

train_grouped_data = train_data.groupby("Sentence_Num_Global").agg({'Token': lambda x: list(x), 'Tag': lambda x: list(x)})
train_grouped_data.reset_index(inplace=True)


validation_dataset = Dataset.from_pandas(dev_grouped_data)
test_dataset = Dataset.from_pandas(test_grouped_data)
train_dataset = Dataset.from_pandas(train_grouped_data)

raredis_datasets = DatasetDict({
    'train': train_dataset, 
    'validation': validation_dataset,
    'test': test_dataset
})


all_tags = set()
for dataset in [train_dataset, validation_dataset, test_dataset]:
    all_tags.update([tag for sentence_tags in dataset['Tag'] for tag in sentence_tags])
all_tags = sorted(list(all_tags))  # Sort for consistency


tag_feature = ClassLabel(names=all_tags)


features = Features({
    'Sentence_Num_Global': Value('int32'),
    'Token': Sequence(Value('string')),
    'Tag': Sequence(tag_feature)
})


raredis_datasets.set_format(type='pandas')
for split in raredis_datasets.keys():
    df = raredis_datasets[split].to_pandas()
    df['Tag'] = df['Tag'].apply(lambda tags: [tag_feature.str2int(tag) for tag in tags])
    raredis_datasets[split] = Dataset.from_pandas(df, features=features)



for split in raredis_datasets.keys():
    raredis_datasets[split] = raredis_datasets[split].cast(features)



label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)






## train
#model_checkpoint = "../PLMs/bert-base-cased"
#model_checkpoint = "../PLMs/Bio_ClinicalBERT"
## evaluate
model_checkpoint = 'BERT_finetuned_ner_checkpoints/checkpoint-7074' #bert-base-cased best epoch on validation set

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = evaluate.load("seqeval")
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)





tokenized_datasets = raredis_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raredis_datasets["train"].column_names,
)

print(tokenized_datasets)


args = TrainingArguments(
    "output_finetuned_ner_checkpoints", #output_dir
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=20,
    weight_decay=0.01,
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
#trainer.train()


raw_predictions = trainer.predict(tokenized_datasets["test"])  ## when eval, can change test dataset here, or in the trainer eval_dataset
#print(eval_results)
#print(len(raw_predictions))
predictions = np.argmax(raw_predictions.predictions, axis=2)
#print(len(predictions))
predictions = [
    [pred for pred, label in zip(prediction, label_seq) if label != -100]
    for prediction, label_seq in zip(predictions, raw_predictions.label_ids)
]

predictions_str = [[id2label[label] for label in pred] for pred in predictions]
#print(len(predictions_str))
true_labels = [tokenized_datasets["test"][i]["labels"] for i in range(len(tokenized_datasets["test"]))]

true_labels_str = [[id2label[label] for label in label_seq if label != -100] for label_seq in true_labels]
#print(true_labels_str) ## a list of lists
report = classification_report(true_labels_str, predictions_str, digits=4)
print(report)