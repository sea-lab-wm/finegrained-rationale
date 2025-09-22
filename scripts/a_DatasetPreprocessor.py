import pandas as pd
import chardet

from utils.consts import *
from utils.functions import *


def remove_non_atomic_commits(input_path, output_path):
    with open(input_path, 'rb') as f:
        result = chardet.detect(f.read())

    data = pd.read_csv(input_path, encoding=result['encoding'])

    # Filter out non-atomic commits
    atomic_commits = data[data['if_mulit_commit']!='1']

    # Drop Columns that are not needed
    if 'Unnamed: 9' in atomic_commits.columns and 'if_mulit_commit' in atomic_commits.columns:
        atomic_commits = atomic_commits.drop(columns=['if_mulit_commit', 'Unnamed: 9'])

    # Save the filtered data to a new CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    atomic_commits.to_csv(output_path, index=False)
    print('Successfully removed non-atomic commits to ', output_path)


def convert_csv_to_json(input_path, output_path):
    data = pd.read_csv(input_path)

    data = data[data['sampling_commit']==1]

    data.to_json(output_path, orient='records', indent=4)
    print(f'Successfully converted CSV to pretty JSON and saved to {output_path}')


def get_change_count(row):
    owner, repo, sha = extract_info_from_commit_message(row['url'])

    commit_info = get_commit_info(owner, repo, sha)

    line_change_count = commit_info['stats']['total'] if 'stats' in commit_info and 'total' in commit_info['stats'] else 0
    java_file_change_count = sum(1 for file in commit_info.get('files', []) if file['filename'].endswith('.java'))

    return java_file_change_count, line_change_count

def collect_commit_data(input_path, output_path):
    data = pd.read_csv(input_path)

    for i,row in data.iterrows():
        print(f"Collecting Commit Info: {i + 1}/{len(data)}")
        java_file_change_count, line_change_count = get_change_count(row)
        data.at[i, 'java_file_change_count'] = java_file_change_count
        data.at[i, 'line_change_count'] = line_change_count

    data.to_csv(output_path, index=False)
    return data

def filter_commits(input_path, output_path):

    data = collect_commit_data(input_path, pop_data_with_commit_info)

    IQR_line_change = data['line_change_count'].quantile(0.75) - data['line_change_count'].quantile(0.25)
    upper_bound_line_change = data['line_change_count'].quantile(0.75) + 1.5 * IQR_line_change

    IQR_java_file = data['java_file_change_count'].quantile(0.75) - data['java_file_change_count'].quantile(0.25)
    upper_bound_java_file = data['java_file_change_count'].quantile(0.75) + 1.5 * IQR_java_file

    data = data[(data['java_file_change_count']>=1.0) &
            (data['java_file_change_count']<=upper_bound_java_file) &
            (data['line_change_count']>=1.0) &
            (data['line_change_count']<=upper_bound_line_change)]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print('Successfully filtered commits and saved to ', output_path)


if __name__ == "__main__":
    print('Removing non-atomic commits from dataset...')
    remove_non_atomic_commits(dataset_path, non_atomic_commits_path)

    print('Filtering commits...')
    filter_commits(non_atomic_commits_path, pop_filtered_data)

    print("Converting to json format...")
    convert_csv_to_json(sample_marked_data, preprocess_data)