from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
import torch
from tqdm import tqdm
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from utils import *



train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'

raredis_datasets = create_raredis_datasets(train_path, dev_path, test_path)

# Now you can access the label names like this
label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)




metric = evaluate.load("seqeval")
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}



true_labels = raredis_datasets['test']['Tag']
true_labels_str = [[id2label[label] for label in label_seq if label != -100] for label_seq in true_labels]



#model_medtuned_finetuned_path = '/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all'
model_medtuned_finetuned_path = '/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all/checkpoint-652'

model = AutoPeftModelForCausalLM.from_pretrained(
    model_medtuned_finetuned_path,
    low_cpu_mem_usage=False,
) 
tokenizer = AutoTokenizer.from_pretrained(model_medtuned_finetuned_path)

# Merge LoRA and base model
merged_model = model.merge_and_unload()



#generator = pipeline('text-generation', model=model_path, torch_dtype=torch.float16, device = 0)  #ori
generator = pipeline('text-generation', model=merged_model, tokenizer = tokenizer, torch_dtype=torch.float16, device = 0)   #lora


#print(generator)
#exit()

max_new_tokens = 128

#finetune
#output_path = 'llama2_medtuned_7b_[all]_ft_result_test'
output_path = 'llama2_medtuned_7b_[all]_ft_652_result_test'








f = open(output_path, 'w')
for entry in tqdm(raredis_datasets['test']):
    #print(sent_list)
    sent_list = entry['Token']
    sent= ' '.join(sent_list)
    prompt = f'''
### Instruction:
Your role involves identifying clinical Named Entities in the text and \
applying the BIO labeling scheme. Utilize the following labels to \
classify each entity: DISEASE: If the entity represents a non-rare disease. \
RAREDISEASE: If the entity denotes a rare disease that affects a small \
percentage of the population. SKINRAREDISEASE: If the entity refers to \
a rare skin disease. SIGN: If the entity corresponds to something found \
during a physical exam or from a laboratory test that shows that a person \
may have a condition or disease. SYMPTOM: If the entity relates to a \
physical or mental problem that a person experiences that may indicate \
a disease or condition; cannot be seen and do not show up on medical tests. \
O: If the entity does not fit into any of the above categories. 
### Input:
{sent}
### Output:
'''

    outputs = generator(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    generated_text = outputs[0]['generated_text'].strip()

    f.write('######\n')
    f.write(generated_text+'\n')
    #print()
f.close()

