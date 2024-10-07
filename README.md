# LLM-based Token-level Named Entity Recognition
 
This repository contains code and resources for large language models (LLMs) in token-level clinical named entity recognition (NER) on rare disease datasets, specifically working with the RareDis dataset.

## Dataset
The original RareDis-v1 dataset should have the following structure. Please refer to the authors of the RareDis dataset for access to the raw data. Once you have obtained the data, you can use brat2iob.py to convert the raw data (BRAT format) to IOB format. You can also directly use the converted data.
```
RareDis-v1
├── dev                   
├── test               
├── train
    ├── Abetalipoproteinemia.ann
    ├── Abetalipoproteinemia.txt
    └── ...         
└── README.md
```

## Repository Structure

- **brat2iob.py**: A script to convert raw data from BRAT format to IOB format.
- **rd_ner_eval.py**: Evaluation script for the model outputs.
- **rd_ner_ft.py**: Fine-tuning script for encoder-only models (BERT, etc.) on rare disease data.
- **rd_ner_llm.py**: A script leveraging large language models (LLMs) for NER on the RareDis dataset.
- **rd_ner_llm_finetune.py**: Fine-tuning script specifically for LLMs (Llama-2-7b , etc.) on NER tasks.
- **rd_ner_openai.py**: A script using OpenAI models for NER tasks on RareDis.
- **rd_ner_rag.py**: A script for implementing Retrieval-Augmented Generation (RAG) with NER tasks on RareDis.
- **gpt4-1106-preview-chat_[all]_5shot_result_test**: Example output of NER predictions using a 5-shot GPT-4 model.
- **utils.py**: Utility functions to support the main scripts.
- **requirements.txt**: A list of Python dependencies required to run the code in this repository.

## Installation

To run the scripts in this repository, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Example
You can find an example of model output in **gpt4-1106-preview-chat_[all]_5shot_result_test**, showcasing predictions from a GPT-4 model with 5-shot learning.


## References

If you find this code useful, please consider citing our works:

```bibtex
@article{lu2024large,
  title={Large Language Models Struggle in Token-Level Clinical Named Entity Recognition},
  author={Lu, Qiuhao and Li, Rui and Wen, Andrew and Wang, Jinlian and Wang, Liwei and Liu, Hongfang},
  journal={arXiv preprint arXiv:2407.00731},
  year={2024}
}
```



