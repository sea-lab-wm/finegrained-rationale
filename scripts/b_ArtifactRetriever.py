import pandas as pd
import concurrent.futures
from tqdm import tqdm

from utils.consts import *
from utils.functions import *



def collect_commit_info(input_path, output_path):
    print(f"Collecting commit info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding commit info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        print(f"Processing commit {i + 1}/{len(commits)}: {commit['url']}")
        owner, repo, sha = extract_info_from_commit_message(commit['url'])

        commit_info = get_commit_info(owner, repo, sha)

        commit['commit_info'] = commit_info

    print(f"Writing commit info to {output_path}...")
    write_json(commits, output_path)


def extract_ids_from_list_str(list_str):
    issue_numbers = list(map(int, re.findall(r'\d+', list_str)))

    if len(issue_numbers) == 0:
        return []

    return issue_numbers


def collect_issue_info(input_path, output_path):
    print(f"Collecting issue info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding issue info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        print(f"Collecting issue info for commit {i + 1}/{len(commits)}: {commit['url']}")
        owner, repo, _ = extract_info_from_commit_message(commit['url'])
        issue_ids = commit['issue_id']

        issue_info_all = []
        for issue_id in issue_ids:
            issue_info = get_issue_info(owner, repo, issue_id)

            # wait for a while to avoid rate limit
            time.sleep(0.01)

            if issue_info:

                comments = []
                if issue_info['comments'] > 0:
                    comments = get_comments_from_link(issue_info.get('comments_url', ''))

                    # wait for a while to avoid rate limit
                    time.sleep(0.01)

                issue_info['comments_details'] = comments
                issue_info_all.append(issue_info)

        commit['issue_info'] = issue_info_all

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)


def collect_pr_info(input_path, output_path):
    print(f"Collecting PR info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding pr info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        print(f"Collecting pr info for commit {i + 1}/{len(commits)}: {commit['url']}")
        owner, repo, _ = extract_info_from_commit_message(commit['url'])
        ids = commit['pr_id']

        pr_info_all = []
        for id in ids:
            pr_info = get_pr_info(owner, repo, id)

            # wait for a while
            time.sleep(0.01)

            if pr_info:

                comments = []
                if pr_info['comments'] > 0:
                    comments = get_comments_from_link(pr_info.get('comments_url', ''))

                    # wait for a while
                    time.sleep(0.01)

                pr_info['comments_details'] = comments

                pr_info_all.append(pr_info)

        commit['pr_info'] = pr_info_all

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)


def collect_cr_info(input_path, output_path):
    print(f"Collecting CR comments info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding pr info to {len(commits)} commits...")
    for i, commit in enumerate(commits):
        print(f"Collecting cr comment for commit {i + 1}/{len(commits)}: {commit['url']}")

        for pr_info in commit.get('pr_info',[]):

            comments = []
            if pr_info['review_comments'] > 0:

                comments = get_comments_from_link(pr_info.get('review_comments_url', ''))

                # wait for a while
                time.sleep(0.01)

            pr_info['review_comments_details'] = comments

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)

def collect_javadoc_info(input_path, output_path):
    print(f"Collecting commit info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding code comment info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        print(f"Processing commit {i + 1}/{len(commits)}: {commit['url']}")

        for file_info in commit['commit_info']['files']:
            added_texts, removed_texts = get_comments_from_code_change(file_info)

            file_info['added_comments'] = added_texts
            file_info['removed_comments'] = removed_texts

    print(f"Adding javadocs info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        print(f"Processing commit {i + 1}/{len(commits)}: {commit['url']}")

        for file_info in commit['commit_info']['files']:
            class_docstrings, method_docstrings = get_docstring_info(file_info)

            file_info['class_docstrings'] = class_docstrings
            file_info['method_docstrings'] = method_docstrings

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)

def collect_ref_info(input_path, output_path):
    print(f"Collecting Commit info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding reference info to {len(commits)} commits...")
    for i, commit in enumerate(commits):

        try:
            print(f"Collecting reference for commit {i + 1}/{len(commits)}: {commit['url']}")

            owner, repo, sha = extract_info_from_commit_message(commit['url'])

            issue_ids = extract_ids_from_list_str(commit['issue_id'])
            pr_ids = extract_ids_from_list_str(commit['pr_id'])

            issue_list, pr_list = get_ref_IssueOrPR(owner, repo, sha, issue_ids + pr_ids)

            commit['ref_pr_info'] = pr_list
            commit['ref_issue_info'] = issue_list

            if i % 30 == 0:
                print(f"Sleeping for 60 seconds to avoid rate limit...")
                time.sleep(60)
            else:
                time.sleep(.1)
        except Exception as e:
            print(e)
            print('*' * 50)
            print('*' * 50)
            print(f"Error in commit {i + 1}/{len(commits)}: {commit['url']}")

            print(f"Writing issue info to {output_path}...")
            write_json(commits, output_path)
            print('*' * 50)
            print('*' * 50)
            break

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)

def collect_additional_ref_info(input_path, output_path):
    print(f"Collecting Commit info from {input_path}...")
    commits = read_json(input_path)

    print(f"Adding reference info to {len(commits)} commits...")
    for i, commit in enumerate(commits):
        try:
            if (i+1) < 250: # TODO
                continue

            print(f"Collecting reference for commit {i + 1}/{len(commits)}: {commit['url']}")

            owner, repo, sha = extract_info_from_commit_message(commit['url'])

            existing_ids = set()

            issue_ids = extract_ids_from_list_str(commit['issue_id'])
            pr_ids = extract_ids_from_list_str(commit['pr_id'])
            ref_issue_ids = [issue['number'] for issue in commit['ref_issue_info']]
            ref_pr_ids = [issue['number'] for issue in commit['ref_pr_info']]

            existing_ids.update(issue_ids+pr_ids+ref_issue_ids+ref_pr_ids)

            texts = []

            texts.extend(get_texts_from_issue_pr(commit, 'issue_info'))
            texts.extend(get_texts_from_issue_pr(commit, 'pr_info'))
            texts.extend(get_texts_from_issue_pr(commit, 'ref_issue_info'))
            texts.extend(get_texts_from_issue_pr(commit, 'ref_pr_info'))

            found_ids = set()
            for text in texts:
                ids = extract_github_ids(text, owner=owner, repo=repo)
                found_ids.update(ids)

            new_ids = found_ids - existing_ids

            issue_list, pr_list = get_IssueOrPR(owner, repo, new_ids)

            commit['add_ref_pr_info'] = pr_list
            commit['add_ref_issue_info'] = issue_list

            time.sleep(0.1)

            if (i+1) % 10 == 0:
                print(f"Writing issue info to {output_path}...")
                write_json(commits, output_path)

        except Exception as e:
            print(e)
            print('*' * 50)
            print(f"Error in commit {i + 1}/{len(commits)}: {commit['url']}")
            print('*' * 50)
            break

    print(f"Writing issue info to {output_path}...")
    write_json(commits, output_path)


def get_texts_from_issue_pr(commit, column_name):
    texts = []
    for issue in commit.get(column_name, []):
        texts.append(issue['title'])
        texts.append(issue['body'])
        for comment in issue['comments_details']:
            texts.append(comment['body'])
        if "pr_info" in column_name:
            for comment in issue['review_comments_details']:
                texts.append(comment['body'])
    return texts


def fix_commits_with_many_comments(input_path):
    commits = read_json(input_path)

    for i, commit in enumerate(commits):
        print(f"Processing commit {i + 1}/{len(commits)}: {commit['url']}")

        for issue in commit['issue_info']:
            if issue['comments'] > 30:
                comments = get_comments_from_link(issue.get('comments_url', ''))

                if len(comments) != issue['comments']:
                    raise Exception(f"Number of comments does not match the number of comments in issue {issue['url']}")
                else:
                    issue['comments_details'] = comments

        if "pr_info" in commit:
            for issue in commit['pr_info']:
                if issue['comments'] > 30:
                    comments = get_comments_from_link(issue.get('comments_url', ''))

                    if len(comments) != issue['comments']:
                        raise Exception(
                            f"Number of comments does not match the number of comments in issue {issue['url']}")
                    else:
                        issue['comments_details'] = comments

                if issue['review_comments'] > 30:
                    comments = get_comments_from_link(issue.get('review_comments_url', ''))

                    if len(comments) != issue['review_comments']:
                        raise Exception(
                            f"Number of comments does not match the number of comments in issue {issue['url']}")
                    else:
                        issue['review_comments_details'] = comments

    print(f"Writing issue info to {input_path}...")
    write_json(commits, input_path)

def filter_issue_body(text):
    if text is None:
        return ""
    elif text.startswith("## What is the purpose of the change"):
        text = text.replace("\r", "")
        text = "\n".join([t for t in text.split("\n") if t and not t.startswith("#")])

        idx = text.find("Follow this checklist to help us incorporate your contribution quickly and easily")
        text = text[:idx]

        return text
    else:
        return text

def process_commit(commit):

    # parse commit message
    commit["commit_info"]["commit"]["parsed_message"] = parse_sentences_from_text(commit["commit_info"]["commit"]['message'])

    # parse added code comments
    comment_types = ["added_comments", "removed_comments"]
    for comment_type in comment_types:
        for file in commit["commit_info"]["files"]:
            comments = []
            for comment in file.get(comment_type, []):
                sents = parse_sentences_from_text(comment)
                comments.append({
                    "text": comment,
                    "sentences": sents
                })
            file[f"parsed_{comment_type}"] = comments

    # parse docstrings
    javadocs_types = ["class_docstrings", "method_docstrings"]
    for javadocs_type in javadocs_types:
        for file in commit["commit_info"]["files"]:
            javadocs = []
            javadoc_objects = file[javadocs_type] if file[javadocs_type] else []
            for javadoc_object in javadoc_objects:

                class_name = javadoc_object[0]
                javadoc = javadoc_object[1]

                sents = parse_sentences_from_text(javadoc)
                javadocs.append({
                    "class_name": class_name,
                    "text": javadoc,
                    "sentences": sents
                })
            file[f"parsed_{javadocs_type}"] = javadocs

    # parse issue/pull request
    issue_types = ["issue_info", "pr_info", "ref_pr_info", "ref_issue_info", "add_ref_pr_info", "add_ref_issue_info"]
    for issue_type in issue_types:
        for issue in commit.get(issue_type, []):
            issue["parsed_title"] = parse_sentences_from_text(issue["title"])

            # Filter pull body
            issue_body = filter_issue_body(issue["body"])

            issue["parsed_body"] = parse_sentences_from_text(issue_body)

            for comment in issue.get("comments_details", []):

                # Filter comment in pr
                if comment["body"].startswith("# [Codecov]"):
                    continue

                comment["parsed_body"] = parse_sentences_from_text(comment["body"])

            if "pr_info" in issue_types:
                for comment in issue.get("review_comments_details", []):
                    comment["parsed_body"] = parse_sentences_from_text(comment["body"])
    
    return commit


def parse_sentence(input_path, output_path, num_workers=None):
    commits = read_json(input_path)

    # print(f"Processing {len(next_commits)} commits using multiprocessing...")
    print(f"Processing {len(commits)} commits using multiprocessing...")

    # Use all available CPUs by default
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # tqdm for progress bar (optional)
        # results = list(tqdm(executor.map(process_commit, next_commits), total=len(next_commits)))
        results = list(tqdm(executor.map(process_commit, commits), total=len(commits)))

    # final_commits = previous_commits + results

    print(f"Saving dataset (sentence parsed) to {output_path}")
    # write_json(final_commits, output_path)
    write_json(results, output_path)

def add_issue_pr_ids(input_path, output_path):
    data = read_json(input_path)

    def extract_ids(row):
        owner, repo, _ = extract_info_from_commit_url(row['url'])
        return pd.Series(extract_issue_pr_ids(row['message'], owner=owner, repo=repo))

    # Extract issue and PR IDs from commit messages
    print('Collecting issue and PR IDs from commit messages...')
    for commit in data:
        owner, repo, _ = extract_info_from_commit_message(commit['url'])
        issue_ids, pr_ids = extract_issue_pr_ids(commit['commit_info']['commit']['message'], owner=owner, repo=repo)
        commit['issue_id'] = issue_ids
        commit['pr_id'] = pr_ids

    print('Successfully extracted issue and PR IDs from commit messages.')
    write_json(data, output_path)


if __name__ == '__main__':

    print(f"Adding Commit Info ...")
    collect_commit_info(preprocess_data, dataset_with_commit_info)

    print('Adding issue and PR IDs to commits...')
    add_issue_pr_ids(dataset_with_commit_info, dataset_with_commit_info)
    
    print(f"Adding Issue Info ...")
    collect_issue_info(dataset_with_commit_info, dataset_with_commit_issue_info)
    
    print(f"Adding Pull Request Info ...")
    collect_pr_info(dataset_with_commit_issue_info, dataset_with_commit_issue_pr_info)
    
    print(f"Adding Code Review Comments in Pull Request  ...")
    collect_cr_info(dataset_with_commit_issue_pr_info, dataset_with_commit_issue_pr_cr_info)

    print(f"Adding Code Comments & Javadocs in Commits  ...")
    collect_javadoc_info(dataset_with_commit_issue_pr_cr_info, dataset_with_commit_issue_pr_cr_javadoc_info)

    print(f"Adding reference pr/issue in Commits  ...")
    collect_ref_info(dataset_with_commit_issue_pr_cr_javadoc_info, dataset_with_commit_issue_pr_cr_javadoc_ref_info)
    
    print(f"Adding additional reference pr/issue in Commits  ...")
    collect_additional_ref_info(dataset_with_commit_issue_pr_cr_javadoc_ref_info, dataset_with_commit_issue_pr_cr_javadoc_add_ref_info)

    print(f"Parsing sentences ...")
    parse_sentence(dataset_with_commit_issue_pr_cr_javadoc_add_ref_info, dataset_sentence, os.cpu_count())