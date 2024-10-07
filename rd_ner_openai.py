import logging
import sys
import os
from openai import AzureOpenAI
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value
from tqdm import tqdm
from utils import *

os.environ["AZURE_OPENAI_KEY"] = 'xxx'
os.environ["AZURE_OPENAI_ENDPOINT"] = 'xxx'


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2023-07-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
#deployment_name = 'gpt-35-turbo-1106' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
deployment_name = 'gpt4-1106-preview-chat'


#load raredis data

##data
train_path = 'RareDis-v1/train_combined_IOB_data.csv' 
dev_path = 'RareDis-v1/dev_combined_IOB_data.csv'
test_path = 'RareDis-v1/test_combined_IOB_data.csv'




# Load your CSV file
dev_data = pd.read_csv(dev_path)
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(train_path)

raredis_datasets = create_raredis_datasets(train_path, dev_path, test_path)

label_names = raredis_datasets['train'].features['Tag'].feature.names
print(label_names)


#exit()


#print(raredis_datasets['test'][82])
#exit()

#disease
    ### Instruction:
    #In the provided text, your objective is to recognize and label Named \
    #Entities associated with diseases using the BIO labeling scheme. \
    #Do not include rare diseases that affect a small percentage of the population. \
    #Start by marking the beginning of a disease-related phrase with B \
    #(Begin), and then continue with I (Inner) for the subsequent words \
    #within that phrase. Non-disease words should be labeled as O. \
    #For each input token provided, generate a corresponding label. \
    #Ensure that each label is presented on a separate line, directly aligned with its respective input token. 

#rare disease
    ### Instruction:
    #In the provided text, your objective is to recognize and label Named \
    #Entities associated with rare diseases using the BIO labeling scheme. \
    #A rare disease is a disease that affects a small percentage of the population. \
    #Start by marking the beginning of a rare-disease-related phrase with B \
    #(Begin), and then continue with I (Inner) for the subsequent words \
    #within that phrase. Non-rare-disease words should be labeled as O. \
    #For each input token provided, generate a corresponding label. \
    #Ensure that each label is presented on a separate line, directly aligned with its respective input token.

#skinraredisease
#In the provided text, your objective is to recognize and label Named
#Entities associated with rare diseases using the BIO labeling scheme. 
#A skin rare disease is a skin disease that affects a small percentage of the population.
#Start by marking the beginning of a skin-rare-disease-related phrase with B
#(Begin), and then continue with I (Inner) for the subsequent words
#within that phrase. Non-skin-rare-disease words should be labeled as O.

#sign
    #In the provided text, your objective is to recognize and label Named \
    #Entities associated with signs using the BIO labeling scheme. \
    #A sign is something found during a physical exam or from a laboratory test \
    #that shows that a person may have a condition or disease. \
    #Start by marking the beginning of a sign-related phrase with B \
    #(Begin), and then continue with I (Inner) for the subsequent words \
    #within that phrase. Non-sign words should be labeled as O. \
    #For each input token provided, generate a corresponding label. \
    #Ensure that each label is presented on a separate line, directly aligned with its respective input token. 

#symptom
    #In the provided text, your objective is to recognize and label Named \
    #Entities associated with symptoms using the BIO labeling scheme. \
    #A symptom is a physical or mental problem that a person experiences \
    #that may indicate a disease or condition; cannot be seen and do not show up on medical tests. \
    #Start by marking the beginning of a symptom-related phrase with B \
    #(Begin), and then continue with I (Inner) for the subsequent words \
    #within that phrase. Non-symptom words should be labeled as O. \
    #For each input token provided, generate a corresponding label. \
    #Ensure that each label is presented on a separate line, directly aligned with its respective input token.

#all
#Your role involves identifying clinical Named Entities in the text and \
#applying the BIO labeling scheme. Utilize the following labels to \
#classify each entity: DISEASE: If the entity represents a non-rare disease. \
#RAREDISEASE: If the entity denotes a rare disease that affects a small \
#percentage of the population. SKINRAREDISEASE: If the entity refers to \
#a rare skin disease. SIGN: If the entity corresponds to something found \
#during a physical exam or from a laboratory test that shows that a person \
#may have a condition or disease. SYMPTOM: If the entity relates to a \
#physical or mental problem that a person experiences that may indicate \
#a disease or condition; cannot be seen and do not show up on medical tests. \
#O: If the entity does not fit into any of the above categories. For each \
#input token provided, generate a corresponding label. Ensure that each \
#label is presented on a separate line, directly aligned with its respective input token.



### samples
sent1 = '''
ACTH
deficiency
arises
as
a
result
of
decreased
or
absent
production
of
adrenocorticotropic
hormone
(
ACTH
)
by
the
pituitary
gland
.
'''
output1 = '''
### Output:
ACTH : B-RAREDISEASE
deficiency : I-RAREDISEASE
arises : O
as : O
a : O
result : O
of : O
decreased : O
or : O
absent : O
production : O
of : O
adrenocorticotropic : O
hormone : O
( : O
ACTH : O
) : O
by : O
the : O
pituitary : O
gland : O
. : O
'''
sent2 = '''
A
decline
in
the
concentration
of
ACTH
in
the
blood
leads
to
a
reduction
in
the
secretion
of
adrenal
hormones
,
resulting
in
adrenal
insufficiency
(
hypoadrenalism
)
.
'''
output2 = '''
### Output:
A : O
decline : B-SIGN
in : I-SIGN
the : I-SIGN
concentration : I-SIGN
of : I-SIGN
ACTH : I-SIGN
in : I-SIGN
the : I-SIGN
blood : I-SIGN
leads : O
to : O
a : O
reduction : B-SIGN
in : I-SIGN
the : I-SIGN
secretion : I-SIGN
of : I-SIGN
adrenal : I-SIGN
hormones : I-SIGN
, : O
resulting : O
in : O
adrenal : O
insufficiency : O
( : O
hypoadrenalism : B-DISEASE
) : O
. : O
'''
sent3 = '''
Adrenal
insufficiency
leads
to
weight
loss
,
lack
of
appetite
(
anorexia
)
,
weakness
,
nausea
,
vomiting
,
and
low
blood
pressure
(
hypotension
)
.
'''
output3 = '''
### Output:
Adrenal : B-DISEASE
insufficiency : I-DISEASE
leads : O
to : O
weight : B-SIGN
loss : I-SIGN
, : O
lack : O
of : O
appetite : O
( : O
anorexia : B-SYMPTOM
) : O
, : O
weakness : B-SYMPTOM
, : O
nausea : B-SYMPTOM
, : O
vomiting : B-SIGN
, : O
and : O
low : O
blood : O
pressure : O
( : O
hypotension : B-SIGN
) : O
. : O
'''
sent4 = '''
Because
these
symptoms
are
so
general
,
the
diagnosis
is
sometimes
delayed
or
missed
entirely
.
'''
output4 = '''
### Output:
Because : O
these : O
symptoms : O
are : O
so : O
general : O
, : O
the : O
diagnosis : O
is : O
sometimes : O
delayed : O
or : O
missed : O
entirely : O
. : O
'''
sent5 = '''
When
ACTH
deficiency
is
suspected
,
blood
samples
are
taken
for
analysis
,
especially
of
the
level
of
cortisol
in
the
blood
.
'''
output5 = '''
### Output:
When : O
ACTH : B-RAREDISEASE
deficiency : I-RAREDISEASE
is : O
suspected : O
, : O
blood : O
samples : O
are : O
taken : O
for : O
analysis : O
, : O
especially : O
of : O
the : O
level : O
of : O
cortisol : O
in : O
the : O
blood : O
. : O
'''


#zero shot
#output_path = deployment_name + '_[rare disease]_result_test'
#output_path = deployment_name + '_[disease]_result_test'
#output_path = deployment_name + '_[sign]_result_test'
#output_path = deployment_name + '_[symptom]_result_test'
#output_path = deployment_name + '_[skin raredisease]_result_test'
#output_path = deployment_name + '_[all]_result_test'

#few shot
output_path = deployment_name + '_[all]_5shot_result_test'

error_samples = []  

f = open(output_path, 'w')
idx = 0
for entry in tqdm(raredis_datasets['test']):
    sent_list = entry['Token']
    sent= '\n'.join(sent_list)
    try:
        response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        messages=[
            {"role": "system", "content":'''
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
output is presented on a separate line, in the format of [input token : label] '''},
            ## zero shot
            {"role": "user", "content": "### Input:\n" + sent},
            {"role": "assistant", "content": "### Output:\n"}
            ## 5 shot
            #{"role": "user", "content": "### Input:\n" + sent1},
            #{"role": "assistant", "content": output1},
            #{"role": "user", "content": "### Input:\n" + sent2},
            #{"role": "assistant", "content": output2},
            #{"role": "user", "content": "### Input:\n" + sent3},
            #{"role": "assistant", "content": output3},
            #{"role": "user", "content": "### Input:\n" + sent4},
            #{"role": "assistant", "content": output4},
            #{"role": "user", "content": "### Input:\n" + sent5},
            #{"role": "assistant", "content": output5},
            #{"role": "user", "content": "### Input:\n" + sent},
            #{"role": "assistant", "content": "### Output:\n"}
        ]
    )
        #print(response)
        answer = response.choices[0].message.content
    except Exception as e:
        error_samples.append(idx)
        error_response = '\n'.join(['O' for _ in sent.split('\n')])
        answer = error_response
        answer = '\n'.join([f"{t} : {ta}" for t, ta in zip(sent.split('\n'), error_response.split('\n'))])
        print('error samples: ',error_samples)

    f.write('######\n')
    f.write(str(answer).strip()+'\n')  ## should add strip()
    #print()
    idx += 1
f.close()


