from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, set_global_service_context, PromptTemplate
import logging
import sys
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
import time
from utils import *

## local medtuned
'''
#system_prompt = """[INST] <<SYS>>\n You are a helpful assistant in biomedical informatics.\n<</SYS>>\n\n"""
system_prompt = ''
# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<s>[INST] {query_str} [/INST]")
#query_wrapper_prompt = PromptTemplate("{query_str}")

llm = HuggingFaceLLM(
    context_window=4096, #ori 4096
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.7, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="nlpie/Llama2-MedTuned-7b",
    model_name="nlpie/Llama2-MedTuned-7b",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)
'''

### openai

api_key = "xxx"  ## 
azure_endpoint = "xxx"
api_version = "2023-07-01-preview"


# GPT3.5
'''
llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-35-turbo-1106",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
'''

# GPT4

llm = AzureOpenAI(
    model="gpt-4",
    deployment_name="gpt4-1106-preview-chat",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5") ### local embedding

Settings.llm = llm
Settings.embed_model = embed_model



### load index from disk (need to create an index first)
storage_context = StorageContext.from_defaults(persist_dir="nord_files.index")     
index = load_index_from_storage(storage_context)


query_engine = index.as_query_engine()




##data
train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'

raredis_datasets = create_raredis_datasets(train_path, dev_path, test_path)


label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)



#output_path = 'gpt35_rag_[all]_zeroshot_result_test'
output_path = 'gpt4_rag_[all]_zeroshot_result_test111'


f = open(output_path, 'w')
idx = 0
for entry in tqdm(raredis_datasets['test'].select(range(5))):
    #print(entry)
    #exit()
    sent_list = entry['Token']
    #sent= ' '.join(sent_list) ## for llama2-medtuned
    sent= '\n'.join(sent_list)  ## for chatgpt
    query = f'''
### Instruction:
Your role involves identifying clinical Named Entities in the text and \
applying the BIO labeling scheme. Start by marking the beginning of a \
related phrase with B (Begin), and then continue with I (Inner) for the subsequent words \
within that phrase. Utilize the following labels to \
classify each entity: DISEASE: If the entity represents a non-rare disease. \
RAREDISEASE: If the entity denotes a rare disease that affects a small \
percentage of the population. SKINRAREDISEASE: If the entity refers to \
a rare skin disease. SIGN: If the entity corresponds to something found \
during a physical exam or from a laboratory test that shows that a person \
may have a condition or disease. SYMPTOM: If the entity relates to a \
physical or mental problem that a person experiences that may indicate \
a disease or condition; cannot be seen and do not show up on medical tests. \
O: If the entity does not fit into any of the above categories. So the label \
should be one of the following: [B-DISEASE, I-DISEASE, B-RAREDISEASE, I-RAREDISEASE, \
B-SKINRAREDISEASE, I-SKINRAREDISEASE, B-SIGN, I-SIGN, B-SYMPTOM, I-SYMPTOM, O]. \
For each input token provided, generate a corresponding label. Ensure that each \
output is presented on a separate line, in the format of [input token : label]
### Input:
{sent}
### Output:
'''
    answer = query_engine.query(query)
    f.write('######\n')
    f.write(str(answer).strip()+'\n')  ## should add strip()
    #print()
f.close()







