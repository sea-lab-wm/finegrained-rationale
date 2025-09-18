import json
import os
import re
import time

import requests
import spacy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_info_from_commit_url(url):
    match = re.search(r"github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)", url)
    if not match:
        return None, None, None

    owner, repo, sha = match.groups()
    return owner, repo, sha

def extract_info_from_commit_message(url):
    match = re.search(r"github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)", url)
    if not match:
        return None, None, None

    owner, repo, sha = match.groups()
    return owner, repo, sha


def extract_info_from_pull_link(url):
    match = re.search(r"github\.com/([^/]+)/([^/]+)/pull/([0-9]+)", url)
    if not match:
        return None, None, None

    owner, repo, id = match.groups()
    return owner, repo, id


def parse_sentences_from_text(text):
    if text is None:
        return []

    nlp = spacy.load('en_core_web_trf')
    doc = nlp(text)
    return [str(sent) for sent in doc.sents if str(sent)]

def parse_code(annotations):
    annotations = [a.strip() for a in annotations.split(",")]
    return annotations

def save_commit(input_path, id):
    commits = read_json(input_path)

    for _, commit in enumerate(commits):
        if commit['id'] == id:
            write_json(commit, "data/temp.json")
            break

def read_json(input_path):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found... Returning empty list.")
        return []

    with open(input_path, "r") as f:
        commits = json.load(f)
    return commits


def write_json(commits, output_path):
    if not commits:
        print(f"No commits to write to {output_path}.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(commits, f, indent=4)
        print(f"Dataset with commit info saved to {output_path}")


def print_progress(complete, full_length, bar_length=50):
    complete_perchantage = int(complete / full_length * bar_length)
    print("[" + "+" * complete_perchantage + "-" * bar_length + f"] complete: {complete / full_length * 100:.2f}% index: {complete}")

def if_merged(owner, repo, pull_number):

    # GitHub API endpoint to check if a PR has been merged
    api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/merge"

    # Set up headers with authorization
    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    # Make the GET request
    response = requests.get(api_url, headers=headers)

    # Determine if the PR was merged based on the response status code
    if response.status_code == 204:
        return True  # PR was merged
    elif response.status_code == 404:
        return False  # PR was not merged or does not exist
    else:
        raise Exception(f"Error checking PR merge status: {response.status_code} - {response.text}")

def get_IssueOrPR(owner, repo, ids):
    issue_list = []
    pr_list = []

    for id in ids:
        issue_type = "pull" if is_pull_request(owner, repo, id) else "issue"

        if issue_type == 'pull' and not if_merged(owner, repo, id):
            continue

        if issue_type == 'pull':
            issue_info = get_pr_info(owner, repo, id)
        elif is_valid_issue(owner, repo, id):
            issue_info = get_issue_info(owner, repo, id)
        else:
            continue

        # Get commit info for issue/pr
        comments = []
        if issue_info['comments'] > 0:
            comments = get_comments_from_link(issue_info.get('comments_url', ''))

        issue_info['comments_details'] = comments

        # get code review info for only pr
        if issue_type == 'pull':
            comments = []
            if issue_info['review_comments'] > 0:
                comments = get_comments_from_link(issue_info.get('review_comments_url', ''))

            issue_info['review_comments_details'] = comments

        if issue_type == 'pull':
            pr_list.append(issue_info)
        else:
            issue_list.append(issue_info)

    return issue_list, pr_list

def get_ref_IssueOrPR(owner, repo, sha, ids):

    issue_list = []
    pr_list = []

    url = f"https://api.github.com/search/issues?q=repo:{owner}/{repo}+{sha}&per_page=1000"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        issue_data = response.json()

        for issue in issue_data.get('items', []):
            issue_id = issue['number']

            issue_type = "pull" if issue['html_url'].split('/')[-2] == 'pull' else "issue"

            if issue_id in ids:
                continue

            if issue_type == 'pull' and not if_merged(owner, repo, issue_id):
                continue

            if issue_type == 'pull':
                issue_info = get_pr_info(owner, repo, issue_id)
            else:
                issue_info = get_issue_info(owner, repo, issue_id)

            # Get commit info for issue/pr
            comments = []
            if issue_info['comments'] > 0:
                comments = get_comments_from_link(issue_info.get('comments_url', ''))

            issue_info['comments_details'] = comments

            # get code review info for only pr
            if issue_type == 'pull':
                comments = []
                if issue_info['review_comments'] > 0:
                    comments = get_comments_from_link(issue_info.get('review_comments_url', ''))

                issue_info['review_comments_details'] = comments

            if issue_type == 'pull':
                pr_list.append(issue_info)
            else:
                issue_list.append(issue_info)
    else:
        raise Exception(f"Error getting issue/pr references: {response.status_code} - {response.text}")

    return issue_list, pr_list

def calculate_results(y_pred, y_test):
    def float_format(value):
        value *= 100
        value = round(value, 6)
        return value

    accuracy = float_format(accuracy_score(y_test, y_pred))
    precision = float_format(precision_score(y_test, y_pred))
    recall = float_format(recall_score(y_test, y_pred))
    f1 = float_format(f1_score(y_test, y_pred))

    return accuracy, f1, precision, recall

def get_commit_info(owner, repo, sha):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    result = {}

    if response.status_code == 200:
        data = response.json()
        result = data

    return result

def get_comments_from_link(url):
    comments = []
    page = 1

    # Prepare headers, including Authorization only if token is available
    token = os.environ.get("GITHUB_TOKEN")
    headers = {'Accept': 'application/vnd.github.v3+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    while True:
        paged_url = f"{url}?per_page=100&page={page}"
        response = requests.get(paged_url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Error getting comments: {response.status_code} - {response.text}")

        data = response.json()
        if not data:
            break

        comments.extend(data)
        if len(data) < 100:
            break  # No more pages

        page += 1

    return comments

def get_issue_info(owner, repo, issue_id):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_id}"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    result = {}

    if response.status_code == 200:
        data = response.json()
        result = data
    else:
        raise Exception(f"Error getting issue info: {response.status_code} - {response.text}")

    return result

def get_pr_info(owner, repo, id):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{id}"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    result = {}

    if response.status_code == 200:
        data = response.json()
        result = data
    else:
        raise Exception(f"Error checking PR info: {response.status_code} - {response.text}")

    return result

def get_changed_comments(patch_text, sign):
    changes = []
    javadoc_re = re.compile(rf'^([{sign}])\s*\*\s?(.*)')  # + * Javadoc
    inline_re = re.compile(rf'^([{sign}]).*?//\s?(.*)')  # + code…// inline comment
    temp = ''

    for line in patch_text.splitlines():
        for pattern in (javadoc_re, inline_re):
            m = pattern.match(line)
            if m:
                sign, comment = m.groups()
                temp += f"{' ' if temp else ''}{comment}"
                break
        else:
            if temp != '':
                changes.append(temp)
                temp = ''
    if temp != '':
        changes.append(temp)
    return changes

def get_comments_from_code_change(file):
    added_texts = []
    removed_texts = []

    if file['filename'].endswith('.java'):
        patch_text = file.get('patch', '')

        added_changes = get_changed_comments(patch_text, '+')
        removed_changes = get_changed_comments(patch_text, '-')

        added_texts.extend(added_changes)
        removed_texts.extend(removed_changes)

    return added_texts, removed_texts

def get_changed_target_lines(patch: str) -> set[int]:
    changed = set()
    cur_new = None

    for raw in patch.splitlines():
        # When we hit a hunk header, reset cur_new to the new-file start:
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', raw)
        if m:
            cur_new = int(m.group(1))
            continue

        if cur_new is None:
            # skip any lines before the first hunk
            continue

        # skip file markers
        if raw.startswith('+++') or raw.startswith('---'):
            continue

        if raw.startswith('+'):
            # an addition: record this new-file line, then advance
            changed.add(cur_new)
            cur_new += 1
        elif raw.startswith('-'):
            changed.add(cur_new)
        elif raw.startswith(' '):
            cur_new += 1
    return changed

def find_blocks(lines: list[str], signature_regex: str) -> list[tuple[str, int, int]]:
    text = "\n".join(lines)
    blocks = []
    for m in re.finditer(signature_regex, text):
        name = m.group(2)
        start = text[: m.start()].count("\n") + 1  

        depth = 0
        seen_open = False
        for j in range(start - 1, len(lines)):
            line = lines[j]
            # count braces
            opens = line.count("{")
            closes = line.count("}")

            if opens:
                depth += opens
                seen_open = True
            if closes:
                depth -= closes

            # only close the block after we've seen an opening brace
            if seen_open and depth == 0:
                end = j + 1  # 1-based
                blocks.append((name, start, end))
                break

    return blocks

def extract_javadoc_above(lines: list[str], start_line: int, changed: set[int]) -> str | None:
    i = start_line - 2  # zero-based index of the line just above the signature
    # skip blank lines AND annotation lines (lines starting with @)
    while i >= 0 and (lines[i].strip() == "" or lines[i].strip().startswith("@")):
        i -= 1

    if i < 0 or not lines[i].strip().endswith("*/"):
        return None

    # gather the /** … */ lines
    doc_lines = []
    while i >= 0:

        doc_lines.insert(0, lines[i])
        if lines[i].strip().startswith("/**"):
            break
        i -= 1

    return "\n".join(doc_lines).strip()

def fix_docstring(docstring: str) -> str:
    # 1) Trim and strip the /** … */ markers
    ds = docstring.strip()
    if ds.startswith("/**"):
        ds = ds[3:]
    if ds.endswith("*/"):
        ds = ds[:-2]

    # 2) Split into lines and clean each
    lines = ds.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        # remove a leading '*' if present
        if line.startswith("*"):
            line = line[1:].lstrip()
        # skip blank lines
        if line:
            cleaned.append(line)

    # 3) Merge with spaces
    return " ".join(cleaned)

def get_docstring_info(f):
    class_docstrings = []
    method_docstrings = []

    if not f["filename"].endswith(".java"):
        return None, None

    patch = f.get("patch", "")

    if not patch:
        return None, None

    changed = get_changed_target_lines(patch)
    raw = get_file_content(f["raw_url"])
    lines = raw.splitlines()

    # find every class block
    class_sig = r'\b(class|interface|enum)\s+([A-Za-z_]\w*)'
    classes = find_blocks(lines, class_sig)

    # find every method block
    method_sig = (
        r'\b(public|protected|private|static)\b'  # visibility
        r'.*?\b([A-Za-z_]\w*)\s*\([^)]*\)\s*{'  # name(…) {
    )
    methods = find_blocks(lines, method_sig)

    # for each class whose block overlaps any changed line, pull its javadoc
    for name, start, end in classes:
        if any(start <= ln <= end for ln in changed):
            doc = extract_javadoc_above(lines, start, changed) or "<no JavaDoc>"
            if doc != "<no JavaDoc>":
                class_docstrings.append((name, fix_docstring(doc)))

    # similarly for methods
    for name, start, end in methods:
        if any(start <= ln <= end for ln in changed):
            doc = extract_javadoc_above(lines, start, changed) or "<no JavaDoc>"
            if doc != "<no JavaDoc>":
                method_docstrings.append((name, fix_docstring(doc)))

    return class_docstrings, method_docstrings


def get_file_content(url):
    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    # wait for a while
    time.sleep(0.01)  

    result = ''

    if response.status_code == 200:
        result = response.text
    else:
        raise RuntimeError(f"Request to {url} returned {response.status_code}")

    return result


def extract_issue_pr_ids(commit_msg, owner=None, repo=None):
    issue_ids = []
    pr_ids = []

    # Extract issue and PR IDs from commit messages
    ids = extract_github_ids(commit_msg, owner=owner, repo=repo)

    # Check if the ID is issue or PR
    for id in ids:
        if is_pull_request(owner, repo, id):
            pr_ids.append(id)
        elif is_valid_issue(owner, repo, id):
            issue_ids.append(id)
        else:
            print(f"ID {id} from owner: {owner}, repo:{repo} is neither a valid issue nor a pull request in {owner}/{repo}")

    return issue_ids, pr_ids

def extract_github_ids(text, owner=None, repo=None):
    if text is None:
        return []

    patterns = [
        r'#(\d+)',
        r'gh.?(\d+)',
        r'issue.?(\d+)',
        r'pull.?(\d+)',
    ]

    if repo and owner:
        patterns.extend([
            rf'{repo}-(\d+)',
            rf'github.com/{owner}/{repo}/issues/(\d+)',
            rf'github.com/{owner}/{repo}/pull/(\d+)',
        ])

    ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        ids.update(matches)
    return sorted(set(map(int, ids)))

def is_pull_request(owner, repo, id) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{id}"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return {}
    else:
        raise Exception(f"Error checking pull request: {response.status_code} - {response.text}")

def is_valid_issue(owner, repo, id) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{id}"

    headers = {
        'Authorization': f'token {os.environ["GITHUB_TOKEN"]}' if os.environ.get('GITHUB_TOKEN', '') else None
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return {}
    else:
        raise Exception(f"Error checking issue: {response.status_code} - {response.text}")