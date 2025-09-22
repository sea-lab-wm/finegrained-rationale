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
    atomic_commits.to_csv(output_path, index=False)
    print('Successfully removed non-atomic commits to ', output_path)


def add_issue_pr_ids(input_path, output_path):
    data = pd.read_csv(input_path)

    def extract_ids(row):
        owner, repo, _ = extract_info_from_commit_url(row['url'])
        return pd.Series(extract_issue_pr_ids(row['message'], owner=owner, repo=repo))

    # Extract issue and PR IDs from commit messages
    print('Collecting issue and PR IDs from commit messages...')
    data[['issue_id', 'pr_id']] = data.apply(extract_ids, axis=1)
    print('Successfully extracted issue and PR IDs from commit messages.')

    # Save the updated DataFrame to a new CSV file
    data.to_csv(output_path, index=False)
    print('Successfully added issue and PR IDs to ', output_path)


def convert_csv_to_json(input_path: str, output_path: str) -> None:
    data = pd.read_csv(input_path)
    data.to_json(output_path, orient='records', indent=4)
    print(f'Successfully converted CSV to pretty JSON and saved to {output_path}')


def filter_commits(input_path, output_path):
    commits = read_json(input_path)

    final_commits = []
    for _, commit in enumerate(commits):

        flag = False
        for file in commit["commit_info"]["files"]:
            if file["filename"].endswith(".java"):
                flag = True
                break

        if flag:
            final_commits.append(commit)

    print(f"Found {len(final_commits)} commits filtered out from {len(commits)} commits.")
    write_json(final_commits, output_path)

if __name__ == "__main__":
    print('Removing non-atomic commits from dataset...')
    remove_non_atomic_commits(dataset_path, non_atomic_commits_path)

    # print('Filtering commits...')
    # filter_commits(dataset_with_commit_issue_pr_cr_javadoc_add_ref_info, dataset_filtered)

    # print('Adding issue and PR IDs to commits...')
    # add_issue_pr_ids(non_atomic_commits_path, issue_pr_id_from_commit_msg)

    # print("Converting to json format...")
    # convert_csv_to_json(issue_pr_id_from_commit_msg, preprocess_data)