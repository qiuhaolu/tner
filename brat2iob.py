import re
from collections import defaultdict
import os
import csv
import string

# File paths for the provided data
#data_folder = 'RareDis-v1/dev'
#data_folder = 'RareDis-v1/test'
data_folder = 'RareDis-v1/train'

ann_files = [f for f in os.listdir(data_folder) if f.endswith('.ann')]
txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

# Verify if the number of annotation files and text files match
if len(ann_files) != len(txt_files):
    raise Exception("The number of annotation files and text files do not match.")

# Sort files to ensure matching pairs
ann_files.sort()
txt_files.sort()

def tokenize_text_with_sentences(text, sentence_num_global):
    """
    Tokenize the text into words, treating punctuation as separate tokens, and track sentence boundaries.
    """
    tokens = []
    current_pos = 0
    sentence_num = 1  # Starting with sentence number 1
    text_length = len(text)
    for word in text.split():
        start = text.find(word, current_pos)

        # Handle punctuation at the start of the word
        while word and word[0] in string.punctuation:
            tokens.append((word[0], start, start + 1, sentence_num))
            word = word[1:]
            start += 1

        end = start + len(word)

        # Temporarily store word tokens if they have punctuation at the end
        word_tokens = []
        while word and word[-1] in string.punctuation:
            word_tokens.insert(0, (word[-1], end - 1, end, sentence_num))
            word = word[:-1]
            end -= 1

        # Add the main word token
        if word:
            tokens.append((word, start, end, sentence_num))

        # Add the stored punctuation tokens at the end
        tokens.extend(word_tokens)

        # Update sentence number if punctuation indicates end of sentence
        if tokens and tokens[-1][0] in {'.', '?', '!'}:
            next_char_index = tokens[-1][2]
            if next_char_index >= text_length or text[next_char_index].isspace():
                sentence_num += 1
                sentence_num_global += 1  # Increment global sentence counter

        current_pos = end + 1


    return tokens, sentence_num_global
 

def annotate_tokens_with_sentences(tokens, entities, sentence_num_global):
    """
    Annotate tokens with IOB tags based on the entity annotations and include sentence information.
    """
    annotated_tokens = []
    for token, start, end, sentence_num in tokens:
        tag = 'O'  # Default tag is 'Outside'
        for entity_type, spans in entities.items():
            for span_start, span_end in spans:
                if start == span_start:
                    tag = f'B-{entity_type}'  # Beginning of an entity
                    break
                elif start > span_start and end <= span_end:
                    tag = f'I-{entity_type}'  # Inside an entity
                    break
        annotated_tokens.append((token, tag, sentence_num, sentence_num_global))
    return annotated_tokens


def parse_ann_file(ann_file_path):
    """
    Parse the .ann file to extract entity annotations.
    Returns a dictionary where keys are entity types and values are lists of (start, end) character positions.
    """
    entities = defaultdict(list)
    with open(ann_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                entity_id, entity_info, _ = parts
                entity_type, start, end = re.split(r'[\s;]+', entity_info)[:3]
                entities[entity_type].append((int(start), int(end)))
    return entities

def tokenize_text(text):
    """
    Tokenize the text into words and return a list of tuples (word, start, end).
    """
    # Using simple whitespace tokenization, can be replaced with more sophisticated tokenizers if needed
    tokens = []
    current_pos = 0
    for word in text.split():
        start = text.find(word, current_pos)
        end = start + len(word)
        tokens.append((word, start, end))
        current_pos = end
    return tokens




# Initialize the global sentence counter at the beginning of the script
sentence_num_all = 0



# Process each file pair and generate IOB formatted data
iob_data = {}
for ann_file, txt_file in zip(ann_files, txt_files):
    ann_file_path = os.path.join(data_folder, ann_file)
    txt_file_path = os.path.join(data_folder, txt_file)

    # Parse the annotation file
    entities = parse_ann_file(ann_file_path)

    # Read and tokenize the text file
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    tokens, sentence_num_all = tokenize_text_with_sentences(text, sentence_num_all)
    # Annotate the tokens
    annotated_tokens = annotate_tokens_with_sentences(tokens, entities, sentence_num_all)

    # Store the IOB data
    iob_data[txt_file] = annotated_tokens

# Define the CSV file path
csv_file_path = os.path.join(data_folder, "combined_IOB_data.csv")
prev_sentence_global = 0
# Create and write data to the CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Writing header
    writer.writerow(['Token', 'Tag', 'Sentence_Num', 'Sentence_Num_Global', 'File'])

    # Writing data
    for file_name, annotated_tokens in iob_data.items():
        for token, tag, sentence_num, sentence_num_all in annotated_tokens:
            writer.writerow([token, tag, sentence_num, prev_sentence_global+sentence_num, file_name.replace('_IOB.txt', '')])
        prev_sentence_global = sentence_num_all



