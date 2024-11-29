from collections import Counter

def check_data_balance(labels_file):
    """
    Check the number of samples in each class.
    :param labels_file: Path to the labels.txt file.
    """
    with open(labels_file, 'r') as file:
        lines = file.readlines()
    
    labels = [line.strip().split()[-1] for line in lines]  # Extract labels
    label_counts = Counter(labels)  # Count occurrences of each label
    
    for label, count in label_counts.items():
        print(f"Class {label}: {count} samples")


# print('raw')
# check_data_balance("/raid/co_show02/JY/CNN/txt/rawlabels.txt")
# print('bal')
# check_data_balance("/raid/co_show02/JY/CNN/txt/balanced_labels.txt")
# print('Final')
check_data_balance("/raid/co_show02/JY/CNN/final_labels.txt")
