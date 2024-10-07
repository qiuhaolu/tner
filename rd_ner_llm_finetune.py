from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
import torch
from utils import *



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





train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'


dev_data = pd.read_csv(dev_path)
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)



raredis_datasets = create_raredis_datasets(train_path, dev_path, test_path)



label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}




for set_name in ['train', 'validation', 'test']:
    rd_set = raredis_datasets[set_name]
    f_input = []
    f_output = []
    
    for data in rd_set:
        tag = data['Tag']
        token = data['Token']
        f_input.append(' '.join(token))
        f_output.append('\n'.join([f"{t} : {id2label[ta]}" for t, ta in zip(token, tag)]))

    # Assuming the method add_column exists in your data structure
    raredis_datasets[set_name] = raredis_datasets[set_name].add_column('f_input', f_input).add_column('f_output', f_output)




def format_instruction(sample):
	return f"""### Instruction:
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
{sample['f_input']}
### Output:
{sample['f_output']}
"""


use_flash_attention = False


## train


model_id = "nlpie/Llama2-MedTuned-7b"  

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"




# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=256,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM", 
)


# prepare model for training
model = prepare_model_for_kbit_training(model)




args = TrainingArguments(
    output_dir="/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all",
    num_train_epochs=10,
    per_device_train_batch_size=6 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False,  # disable tqdm since with packing values are in correct
)


# Upcast layer for flash attnetion
if use_flash_attention:
    from utils.llama_patch import upcast_layer_for_flash_attention
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = upcast_layer_for_flash_attention(model, torch_dtype)

model = get_peft_model(model, peft_config)

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=raredis_datasets['train'],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction, 
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled
# save model
trainer.save_model()




## evaluate
'''
#adapter_output_dir = "/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all"
#adapter_output_dir = "/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all/checkpoint-783"
adapter_output_dir = "/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all/checkpoint-1044"
#adapter_output_dir = "/data_2/qlu1/raredisease/medtuned-7b-int4-raredisease-all/checkpoint-391"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
) 
tokenizer = AutoTokenizer.from_pretrained(adapter_output_dir)

test_sample = raredis_datasets['test'][10]
#print(test_sample)

prompt = f"""### Instruction:
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
{test_sample['f_input']}
### Output:
"""




input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=200, do_sample=True, top_p=0.9,temperature=0.9)

print(f"Prompt:\n{test_sample['f_input']}\n")
print(f"Generated output:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
print(f"Ground truth:\n{test_sample['f_output']}")
'''




### merge and save
'''
model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
) 

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
'''









