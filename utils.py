import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value

def create_raredis_datasets(train_path, dev_path, test_path):
    # Load the CSV files
    dev_data = pd.read_csv(dev_path)
    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)

    # Group the tokens and tags by sentences
    def group_by_sentence(data):
        grouped_data = data.groupby("Sentence_Num_Global").agg({'Token': lambda x: list(x), 'Tag': lambda x: list(x)})
        grouped_data.reset_index(inplace=True)
        return grouped_data

    dev_grouped_data = group_by_sentence(dev_data)
    test_grouped_data = group_by_sentence(test_data)
    train_grouped_data = group_by_sentence(train_data)

    # Convert the grouped data into Huggingface Dataset
    validation_dataset = Dataset.from_pandas(dev_grouped_data)
    test_dataset = Dataset.from_pandas(test_grouped_data)
    train_dataset = Dataset.from_pandas(train_grouped_data)

    # Create a DatasetDict
    raredis_datasets = DatasetDict({
        'train': train_dataset, 
        'validation': validation_dataset,
        'test': test_dataset
    })

    # Step 1: Create a unique list of all tags
    all_tags = set()
    for dataset in [train_dataset, validation_dataset, test_dataset]:
        all_tags.update([tag for sentence_tags in dataset['Tag'] for tag in sentence_tags])
    all_tags = sorted(list(all_tags))  # Sort for consistency

    # Step 2: Create a ClassLabel feature and map each tag to an integer
    tag_feature = ClassLabel(names=all_tags)

    # Define the features of your dataset
    features = Features({
        'Sentence_Num_Global': Value('int32'),
        'Token': Sequence(Value('string')),
        'Tag': Sequence(tag_feature)
    })

    # Apply these features to your datasets
    raredis_datasets.set_format(type='pandas')
    for split in raredis_datasets.keys():
        # Convert the dataset to pandas, update the 'Tag' column, and then convert back to Huggingface dataset
        df = raredis_datasets[split].to_pandas()
        df['Tag'] = df['Tag'].apply(lambda tags: [tag_feature.str2int(tag) for tag in tags])
        raredis_datasets[split] = Dataset.from_pandas(df, features=features)

    # Update the features of your datasets
    for split in raredis_datasets.keys():
        raredis_datasets[split] = raredis_datasets[split].cast(features)

    return raredis_datasets
