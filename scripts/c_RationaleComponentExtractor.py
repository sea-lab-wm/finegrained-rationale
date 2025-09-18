import csv
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm

from utils.consts import *
from utils.functions import *


def get_code_information(template: str, codebook):
    result = []

    for i, row in codebook.iterrows():

        filled = template
        replacements = {
            "<index>": str(i + 1),
            "<code>": row['Annotation Labels'],
            "<definintion>": row['Description'],
            "<question>": row['Component Expressed as Question'],
            # "<importance>": row['Importance'],
            "<rule>": row['Rules'],
            # "<examples>": examples.sample(n=1, random_state=42)['sentence'].iloc[0] if not examples.empty else "No example available.",
        }

        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, str(value))

        result.append(filled)

    result = "\n\n".join(result)

    return result

def get_code_diff_information(template: str, commit_data):
    owner, repo, sha = extract_info_from_commit_message(commit_data['commit_url'])
    commit_info = get_commit_info(owner, repo, sha)

    result = []
    for i, file_info in enumerate(commit_info['files']):
        filled = template
        replacements = {
            "<index>": str(i + 1),
            "<file_name>": file_info['filename'],
            "<diff>": file_info['patch'],
        }

        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)

        result.append(filled)

    result = "\n\n".join(result)
    return result

def get_sentence_information(template: str, commit_data):
    print(commit_data['commit_id'])

    result = {
        "COMMIT_MESSAGE" : [],
        "CLASS_JAVADOCS" : {},
        "METHOD_JAVADOCS" : {},
        "CODE_COMMENT" : [],
        "ISSUE" : {},
        "PULL_REQUEST" : {},
        "CODE_REVIEW_COMMENT" : {}
    }

    for i, sentence in enumerate(commit_data['sentence']):
        sent = sentence['sentence']

        filled = template
        replacements = {
            "<index>": sentence['id'],
            "<source>": sentence['source'],
            "<sentence>": sent,
        }

        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)

        if "JAVADOCS" in sentence['source']:
            class_name = sentence['class_name']

            if class_name not in result[sentence['source']]:
                result[sentence['source']][class_name] = []

            result[sentence['source']][class_name].append(filled)
        elif sentence['source'] in ["ISSUE", "PULL_REQUEST", "CODE_REVIEW_COMMENT"]:
            issue_id = sentence['issue_id']

            if issue_id not in result[sentence['source']]:
                result[sentence['source']][issue_id] = []

            result[sentence['source']][issue_id].append(filled)
        else:
            result[sentence['source']].append(filled)

    final_sentence_info = ""
    for source in result:
        prefix = "\n\nThe following sentences come from "

        if source == "COMMIT_MESSAGE":
            source_placeholder = "the commit message"
            sentences = "\n\n".join(result[source])
            sentences = prefix + source_placeholder + "\n\n" + sentences

            final_sentence_info += sentences
        elif source == "CODE_COMMENT":

            if len(result[source]) > 0:
                source_placeholder = "the comments of the changed code."
                sentences = "\n\n".join(result[source])
                sentences = prefix + source_placeholder + "\n\n" + sentences

                final_sentence_info += sentences
        elif source in ["ISSUE", "PULL_REQUEST", "CODE_REVIEW_COMMENT"]:

            for issue_id in result[source]:
                source_placeholder = f"the {source.lower()}: {issue_id}, associated with the commit"
                sentences = "\n\n".join(result[source][issue_id])
                sentences = prefix + source_placeholder + "\n\n" + sentences

                final_sentence_info += sentences
        elif source in ["CLASS_JAVADOCS", "METHOD_JAVADOCS"]:

            for class_name in result[source]:
                source_placeholder = f"the Javadoc comments of the {source.split('_')[0].lower()}({class_name}) that were changed"
                sentences = "\n\n".join(result[source][class_name])
                sentences = prefix + source_placeholder + "\n\n" + sentences

                final_sentence_info += sentences

    return final_sentence_info

def get_token_count(model_name: str = "gpt-4.1", prompt: str = "") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to base encoding if model not found
        if model_name == "gpt-4.1":
            encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("o200k_base")

    return len(encoding.encode(prompt))

def base_prompt_design(config, dev_path, output_path, test=False):
    print("\n\nStarting base prompt design...")

    experminet_ids = config['exp_ids']
    model_names = config['model_names']

    data = pd.read_csv(dev_path)
    prompt_templates = pd.read_csv("dataset/prompts/prompt_templates.csv")

    data['id'] = data.apply(lambda row: f"{row['commit_id']}_{row['source_id']}_{row['text_id']}_{row['sentence_id']}", axis=1)

    codes = ["GOAL", "NEED", "ALTERNATIVES"]
    codebook = pd.read_csv('dataset/prompts/codebook.csv')
    codebook = codebook[codebook['Annotation Labels'].isin(codes)].reset_index(drop=True)

    data_json = []
    for index, commit_id in enumerate(data['commit_id'].unique()):
        temp_data = {}

        commit_df = data[data['commit_id'] == commit_id].reset_index(drop=True)

        first_row = commit_df.iloc[0]

        temp_data['commit_id'] = str(first_row['commit_id'])
        temp_data['commit_url'] = first_row['commit_url']

        temp_data['sentence'] = []
        for sid, row in commit_df.iterrows():
            if "_JAVADOCS" in row['source']:
                class_part = row['sentence'].split(":")

                class_name = class_part[0].strip().lstrip('[').rstrip(']')
                javadocs = ":".join(class_part[1:]).strip()

                temp_data['sentence'].append({
                    "id": row['id'],
                    "source": row['source'],
                    "class_name": class_name,
                    "sentence": javadocs,
                    "annotations": row['final_annotation'] if pd.notna(row['final_annotation']) else "UNCODED",
                })

            elif ("ISSUE" in row['source']
                  or "PULL_REQUEST" in row['source']
                  or "CODE_REVIEW_COMMENT" in row['source']):
                issue_url_parts = row['source_url'].split('/')

                temp_data['sentence'].append({
                    "id": row['id'],
                    "source": row['source'],
                    "issue_id": "/".join(issue_url_parts[:6]+[issue_url_parts[6].split("#")[0]]),
                    "sentence": row['sentence'],
                    "annotations": row['final_annotation'] if pd.notna(row['final_annotation']) else "UNCODED",
                })

            else:
                temp_data['sentence'].append({
                    "id": row['id'],
                    "source": row['source'],
                    "sentence": row['sentence'],
                    "annotations": row['final_annotation'] if pd.notna(row['final_annotation']) else "UNCODED",
                })

        data_json.append(temp_data)

    for experiment_id in experminet_ids:
        prompt_records = []

        print("+" * 50)
        print(f"Experiment ID: {experiment_id}")
        experiment = experiment_id.split('.')

        rationale_explanation = 0 if experiment[1] == '0' else 1
        rule = 0 if experiment[2] == '0' else 1
        example = 0 if experiment[3] == '0' else f"{experiment[3]}"

        template = prompt_templates[prompt_templates['version'] == experiment_id].iloc[0]

        print(f"Rationale Explanation: {rationale_explanation}, Rules: {rule}, Example: {example}")

        code_information = get_code_information(template['code_information'], codebook)

        for i, commit in enumerate(data_json):
            code_diff_information = get_code_diff_information(template['code_diff_information'], commit)
            sentences_information = get_sentence_information(template['sentences_information'], commit)
            # actual_annotations = [s['annotations'] if s['annotations'] else "UNCODED" for s in commit['sentence']]

            prompt = template['template']
            prompt = prompt.replace("<project_name>", '/'.join(commit['commit_url'].split('/')[3:5]))
            prompt = prompt.replace("<code_information>", code_information)
            prompt = prompt.replace("<code_diff_information>", code_diff_information)
            prompt = prompt.replace("<sentences_information>", sentences_information)
            prompt = prompt.replace("<sentence_count>", str(len(sentences_information)))

            prompt_records.append({
                'rationale_explanation': rationale_explanation,
                'rule': rule,
                'example': example,
                'experiment_version': template['version'],
                'commit_id': commit['commit_id'],
                'prompt': prompt,
            })

        prompt_template_tosend = pd.DataFrame(prompt_records)

        t_output_path = output_path.replace("<exp_id>", f"{'test/' if test else ''}{experiment_id}")

        print(f"Saving prompt templates to {t_output_path}...")
        os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
        prompt_template_tosend.to_csv(t_output_path, index=False)


def get_prompt_response(prompt, model_name):
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_TOKEN") or None)

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt}
            ],
            # temperature=0.1,
            reasoning={
                "effort": "high",
                # 'summary': "auto"
            }
        )

        resp = response.output_text  # direct text output in new API
        return resp.strip() if resp else ""
    except Exception as e:
        print(f"Error generating response for prompt: {e}")
        return ""

def _process_one(args):
    exp_id, model_name, run, input_path, output_path, test = args
    try:
        print(f"[PID {os.getpid()}] Generating responses for experiment {exp_id} "
              f"with model {model_name} ; Run {run}...")

        t_input_path = input_path.replace("<exp_id>", f"{'test/' if test else ''}{exp_id}")
        t_output_path = (
            output_path
            .replace("<exp_id>", f"{'test/' if test else ''}{exp_id}")
            .replace("<model>", model_name)
            .replace("<run>", str(run))
        )

        data = pd.read_csv(t_input_path)

        # per-file processing is sequential inside this process
        data['response'] = data['prompt'].apply(lambda x: get_prompt_response(x, model_name))

        os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
        data.to_csv(t_output_path, index=False)
        msg = f"Responses saved to {t_output_path}."
        print(msg)
        return (exp_id, model_name, run, None, msg)

    except Exception as e:
        err = f"Error processing experiment {exp_id} with model {model_name} ; Run {run}: {e}"
        print(err)
        return (exp_id, model_name, run, err, None)

def get_response_from_prompt(config, input_path, output_path, test=False, max_workers=None):
    exp_ids = config['exp_ids']
    model_names = config['model_names']
    runs = config['runs']

    tasks = [
        (exp_id, model_name, run, input_path, output_path, test)
        for exp_id in exp_ids
        for model_name in model_names
        for run in range(runs)
    ]

    if not tasks:
        print("No tasks to run.")
        return

    if max_workers is None:
        max_workers = max(1, min(mp.cpu_count(), len(tasks)))

    # On some platforms (Windows/macOS), 'spawn' is safer.
    # If you run this inside a Jupyter notebook, consider moving it to a script.
    ctx = mp.get_context("spawn")

    print(f"Starting {len(tasks)} tasks with up to {max_workers} workers...")

    # Use ProcessPoolExecutor for a simple API
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            exp_id, model_name, run, err, msg = fut.result()
            if err:
                # already printed inside worker; reiterate here if you want
                pass
            else:
                # already printed inside worker; keep quiet or log
                pass

    print("All tasks finished.")


def combine_response_labels(row):
    final_annotations = {}

    for code in content_identification_codes:
        final_annotations[code] = 0

    for i in range(3):
        response_label = row.get(f'response_label_{i}', "")
        for code in content_identification_codes:
            if code in response_label:
                final_annotations[code] += 1

    final_annotations = [a for a in final_annotations if final_annotations[a] >= 2]
    return ', '.join(final_annotations)


def calculated_voting_result(response_codes_all_runs):
    result = {
        "code_wise": {}
    }
    tps = fps = tns = fns = 0
    for code in content_identification_codes:
        # use voting mechanism to determine the final response for each code
        # if for a index response are 2 or more are 1 then the final response is 1 else 0
        actual = response_codes_all_runs[code]['actual']
        responses = response_codes_all_runs[code]
        votes = []
        for run in range(3):
            if run in responses:
                votes.append(responses[run])
            else:
                raise Exception(f"Run {run} not found in responses for code {code}")
        votes = pd.DataFrame(votes).T

        votes = votes.apply(lambda x: 1 if x.sum() >= 2 else 0, axis=1)

        tp = ((actual == 1) & (votes == 1)).sum()
        fp = ((actual == 0) & (votes == 1)).sum()
        fn = ((actual == 1) & (votes == 0)).sum()
        tn = ((actual == 0) & (votes == 0)).sum()

        tps += tp
        fps += fp
        tns += tn
        fns += fn

        # calculate precision, recall, f1_score and accuracy
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        accuracy_score = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

        result['code_wise'][code] = [tp, fp, tn, fn, precision_score, recall_score, f1_score, accuracy_score]

    precision_score = tps / (tps + fps) if (tps + fps) > 0 else 0
    recall_score = tps / (tps + fns) if (tps + fns) > 0 else 0
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    accuracy_score = (tps + tns) / (tps + fps + tns + fns) if (tps + fps + tns + fns) > 0 else 0

    result['overall'] = [tps, fps, tns, fns, precision_score, recall_score, f1_score, accuracy_score]

    return result


def save_response_to_development_data(config, development_path, test=False):
    result_path = f"dataset/prompts/component_identification{'/test' if test else ''}/all_result.csv"
    dev_data_path = f"dataset/prompts/component_identification{'/test' if test else ''}/all_{'test' if test else 'dev'}_data.csv"

    exp_ids = config['exp_ids']
    model_names = config['model_names']
    runs = 3

    all_result = []

    development_data = pd.read_csv(development_path)
    development_data['id'] = development_data.apply(lambda row: f"{row['commit_id']}_{row['source_id']}_{row['text_id']}_{row['sentence_id']}", axis=1)

    for exp_id in exp_ids:
        experiment = exp_id.split('.')

        llm_annotation_reasoning = 1 if experiment[1] == '2' else 0
        reasoning_of_example = 0 if experiment[1] == '0' else 1
        rules = 0 if experiment[2] == '0' else 1
        example = 0 if experiment[3] == '0' else f"{experiment[3]}"

        for model_name in model_names:

            precision_scores = []
            recall_scores = []
            f1_scores = []
            accuracy_scores = []
            all_run_tps = []
            all_run_fps = []
            all_run_tns = []
            all_run_fns = []

            response_codes_all_runs = {}
            for code in content_identification_codes:
                response_codes_all_runs[code] = {}

            for run in range(runs):
                print(f"Processing experiment {exp_id}, model {model_name}, run {run}...")

                dev_label_name = f"response_label_{exp_id}_{model_name}_{run}"
                dev_reason_column = f"response_reason_{exp_id}_{model_name}_{run}"

                input_path = f"dataset/prompts/component_identification{'/test' if test else ''}/{exp_id}/{model_name}/{run}/prompt_template_with_response.csv"
                data = pd.read_csv(input_path)
                data = data[data['experiment_version'] == exp_id]

                header = ['sentence_id', 'labels', 'reason'] if llm_annotation_reasoning else ['sentence_id', 'labels']

                response_df = pd.DataFrame([], columns=pd.Series(header))
                for response in data['response']:
                    if isinstance(response, str) and len(response) > 0:
                        response = response.strip()
                    else:
                        continue
                    if not response.startswith('sentence_id,'):
                        response = ','.join(header)+ "\n" + response  # Ensure the header is present

                    reader = csv.reader(io.StringIO(response), skipinitialspace=True)
                    array_of_arrays = [row for row in reader]
                    try:
                        for i, row in enumerate(array_of_arrays):
                            if len(row) > len(header):
                                extra = len(row) - len(header)

                                s = ', '.join(row[-(extra+1):])
                                row = row[:-(extra+1)] + [s]  # Combine extra columns into the last column
                                array_of_arrays[i] = row

                        d = pd.DataFrame(array_of_arrays[1:], columns=array_of_arrays[0])  # Skip the header row
                        response_df = pd.concat([response_df, d], ignore_index=True)
                    except Exception as e:
                        pass

                response_df = response_df[response_df['labels'].notna() & (response_df['labels'] != '')]

                # if there is multiple occurance of sentence_id, merge all the labels
                if len(response_df) != len(response_df['sentence_id'].unique()):
                    # identify the duplicate sentence_ids using value_counts
                    duplicate_sentence_ids = response_df['sentence_id'].value_counts()[response_df['sentence_id'].value_counts() > 1].index.tolist()

                    for d_id in duplicate_sentence_ids:
                        duplicate_labels = response_df[response_df['sentence_id'] == d_id]['labels'].tolist()
                        if 'reason' in response_df.columns:
                            duplicate_reasons = response_df[response_df['sentence_id'] == d_id]['reason'].tolist()
                        annotations = []
                        for l in duplicate_labels:
                            annotations.extend(parse_code(l))
                        annotations = set(annotations)

                        # if for each id in response if it's same as d_id assign its label to ", ".join(annotations)
                        for idx in response_df.index:
                            if response_df.loc[idx, 'sentence_id'] == d_id:
                                response_df.at[idx, 'labels'] = ", ".join(annotations)
                                if 'reason' in response_df.columns:
                                    response_df.at[idx, 'reason'] = "## ".join(duplicate_reasons)

                    response_df = response_df[header].drop_duplicates().reset_index(drop=True)

                response_df = response_df[header].rename(columns={'sentence_id': 'id','labels': dev_label_name, 'reason': dev_reason_column})

                # add label column to development_data on development_data.id = response_df.sentence_id
                development_data = development_data.merge(response_df, on='id', how='left')

                # fill NaN values in response_label with "UNCODED"
                development_data[dev_label_name] = development_data[dev_label_name].fillna('UNCODED')

                tps = fps = tns = fns = 0
                # calculate metrics for the response_label
                for code in content_identification_codes:
                    actual = development_data.apply(lambda row: 1 if code in row['final_annotation'] else 0, axis=1)
                    response = development_data.apply(lambda row: 1 if code in row[dev_label_name] else 0, axis=1)

                    response_codes_all_runs[code]['actual'] = actual
                    response_codes_all_runs[code][run] = response

                    tp = ((actual == 1) & (response == 1)).sum()
                    fp = ((actual == 0) & (response == 1)).sum()
                    fn = ((actual == 1) & (response == 0)).sum()
                    tn = ((actual == 0) & (response == 0)).sum()

                    tps += tp
                    fps += fp
                    tns += tn
                    fns += fn

                    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) \
                        if (precision_score + recall_score) > 0 else 0
                    accuracy_score = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

                    all_result.append([
                        exp_id,
                        llm_annotation_reasoning,
                        reasoning_of_example,
                        rules,
                        example,
                        model_name,
                        run,
                        code,
                        tp,
                        fp,
                        tn,
                        fn,
                        precision_score,
                        recall_score,
                        f1_score,
                        accuracy_score
                    ])

                precision_score = tps / (tps + fps) if (tps + fps) > 0 else 0
                recall_score = tps / (tps + fns) if (tps + fns) > 0 else 0
                f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) \
                    if (precision_score + recall_score) > 0 else 0
                accuracy_score = (tps + tns) / (tps + fps + tns + fns) if (tps + fps + tns + fns) > 0 else 0

                all_run_tps.append(tps)
                all_run_fps.append(fps)
                all_run_tns.append(tns)
                all_run_fns.append(fns)
                precision_scores.append(precision_score)
                recall_scores.append(recall_score)
                f1_scores.append(f1_score)
                accuracy_scores.append(accuracy_score)

                all_result.append([
                    exp_id,
                    llm_annotation_reasoning,
                    reasoning_of_example,
                    rules,
                    example,
                    model_name,
                    run,
                    "Total",
                    tps,
                    fps,
                    tns,
                    fns,
                    precision_score,
                    recall_score,
                    f1_score,
                    accuracy_score
                ])

            voting_result = calculated_voting_result(response_codes_all_runs)
            for code in voting_result['code_wise']:
                all_result.append([
                    exp_id,
                    llm_annotation_reasoning,
                    reasoning_of_example,
                    rules,
                    example,
                    model_name,
                    "Voting",
                    code,
                    voting_result['code_wise'][code][0],  # tp
                    voting_result['code_wise'][code][1],  # fp
                    voting_result['code_wise'][code][2],  # tn
                    voting_result['code_wise'][code][3],  # fn
                    voting_result['code_wise'][code][4],  # precision
                    voting_result['code_wise'][code][5],  # recall
                    voting_result['code_wise'][code][6],  # f1_score
                    voting_result['code_wise'][code][7]   # accuracy_score
                ])

            all_result.append([
                exp_id,
                llm_annotation_reasoning,
                reasoning_of_example,
                rules,
                example,
                model_name,
                "Voting",
                "Total",
                voting_result['overall'][0],  # tp
                voting_result['overall'][1],  # fp
                voting_result['overall'][2],  # tn
                voting_result['overall'][3],  # fn
                voting_result['overall'][4],  # precision
                voting_result['overall'][5],  # recall
                voting_result['overall'][6],  # f1_score
                voting_result['overall'][7]   # accuracy_score
            ])

            all_result.append([
                exp_id,
                llm_annotation_reasoning,
                reasoning_of_example,
                rules,
                example,
                model_name,
                'Average',
                "Total",
                sum(all_run_tps)/len(all_run_tps),
                sum(all_run_fps)/len(all_run_fps),
                sum(all_run_tns)/len(all_run_tns),
                sum(all_run_fns)/len(all_run_fns),
                sum(precision_scores)/len(precision_scores),
                sum(recall_scores)/len(recall_scores),
                sum(f1_scores)/len(f1_scores),
                sum(accuracy_scores)/len(accuracy_scores)
            ])

    all_result = pd.DataFrame(all_result, columns=pd.Series([
        'experiment_version',
        'llm_annotation_reasoning',
        'reasoning_of_example',
        'rules',
        'example',
        'model_name',
        'run',
        'component',
        'tp',
        'fp',
        'tn',
        'fn',
        'precision',
        'recall',
        'f1_score',
        'accuracy_score'
    ]))

    # save the response data
    os.makedirs(os.path.dirname(dev_data_path), exist_ok=True)
    development_data.to_csv(dev_data_path, index=False)
    print(f"Response data saved to {dev_data_path}.")

    # save the all result data
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    all_result.to_csv(result_path, index=False)
    print(f"Results saved to {result_path}.")


def create_development_data(input_path, dev_path, test_path):
    data = pd.read_csv(input_path)

    development_commit_id = data[data['dev_set?']==1]['commit_id'].unique()
    test_commit_id = data[(data['dev_set?']!=1) & (data['example_set?']!=1)]['commit_id'].unique()

    development_data = data[data['commit_id'].isin(development_commit_id)].reset_index(drop=True)
    test_data = data[data['commit_id'].isin(test_commit_id)].reset_index(drop=True)

    # save the development data
    os.makedirs(os.path.dirname(dev_path), exist_ok=True)

    development_data.to_csv(dev_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Development data saved to {dev_path}. Test data saved to {test_path}.")


def calculate_metrice_by_component(input_path, output_path):
    target_experiment = ["1.0.0.0", "1.0.1.0", "1.0.1.1", "1.1.0.0", "1.1.1.0", "1.1.1.1",]
    target_components = ["GOAL", "NEED", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "UNCODED"]

    result = []
    for experiment in target_experiment:
        for component in target_components:
            data_path = input_path.replace("<exp_id>", experiment)

            data = pd.read_csv(data_path)

            data = data[data['final_annotation'].str.contains(component, na=False)].reset_index(drop=True)

            tp = data['tp'].sum()
            fp = data['fp'].sum()
            fn = data['fn'].sum()

            precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                              precision_score + recall_score) > 0 else 0
            result.append([
                experiment,
                component,
                tp,
                fp,
                fn,
                precision_score,
                recall_score,
                f1_score
            ])

            print(f"Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, F1 Score: {f1_score:.4f}")

    # convert the result to a DataFrame and save it
    metrics_df = pd.DataFrame(result, columns=[
        'experiment_version',
        'component',
        'tp',
        'fp',
        'fn',
        'precision',
        'recall',
        'f1_score'
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"Metrics by component saved to {output_path}.")

    result = []
    for experiment in target_experiment:
        data_path = input_path.replace("<exp_id>", experiment)

        data = pd.read_csv(data_path)

        data['tp'] = data.apply(lambda x: calculated_tp(x, False), axis=1)
        data['fp'] = data.apply(lambda x: calculated_fp(x, False), axis=1)
        data['fn'] = data.apply(lambda x: calculated_fn(x, False), axis=1)

        tp = data['tp'].sum()
        fp = data['fp'].sum()
        fn = data['fn'].sum()

        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0

        result.append([
            experiment,
            tp,
            fp,
            fn,
            precision_score,
            recall_score,
            f1_score
        ])

        print(f"Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, F1 Score: {f1_score:.4f}")

        data.to_csv(data_path.replace('.csv', '_uncoded_discarded.csv'), index=False)
        print(f"dataset by experiment saved to {data_path}.")

    # convert the result to a DataFrame and save it
    metrics_df = pd.DataFrame(result, columns=pd.Series([
        'experiment_version',
        'tp',
        'fp',
        'fn',
        'precision',
        'recall',
        'f1_score'
    ]))
    metrics_df.to_csv(output_path.replace('.csv', '_uncoded_discarded.csv'), index=False)
    print(f"Metrics by all components saved to {output_path.replace('.csv', '_all_experiment.csv')}.")


def calculated_metrics(param):
    pass


def calculated_metrics_by_commit(path):
    data = pd.read_csv(path)

    target_components = ["GOAL",	"NEED",	"CONSTRAINTS",	"ALTERNATIVES",	"SELECTED_ALTERNATIVE",	"VALIDATION",	"MATURITY_STAGE",	"SIDE_EFFECTS",	"UNCODED",]
    sources = [
        "COMMIT_MESSAGE",
        "CLASS_JAVADOCS",
        "METHOD_JAVADOCS",
        "CODE_COMMENT",

        "ISSUE",
        # "ISSUE:BODY",
        # "ISSUE: COMMENT",
        # "ISSUE: TITLE",
        # "REF_ISSUE: BODY",
        # "REF_ISSUE: COMMENT",
        # "REF_ISSUE: TITLE",

        "PULL_REQUEST",
        # "PULL_REQUEST: BODY",
        # "PULL_REQUEST: COMMENT",
        # "PULL_REQUEST: TITLE",
        # "REF_PULL_REQUEST: BODY",
        # "REF_PULL_REQUEST: COMMENT",
        # "REF_PULL_REQUEST: TITLE",

        "CODE_REVIEW_COMMENT",
    ]

    result = [
        ["SOURCE"] + target_components
    ]
    for source in sources:
        t = [source]
        for code in target_components:
            t_data = data[(data['source'] == source) & (data['final_annotation'].str.contains(code, na=False))].reset_index(drop=True)
            t.append(t_data['commit_id'].nunique())

        result.append(t)

    metrics_df = pd.DataFrame(result[1:],columns=result[0])

    metrics_df.to_csv('dataset/per_source_code.csv', index=False)
    print(f"Metrics by source and component saved to dataset/per_source_code.csv.")



def calculated_metrics_by_repo(path):
    data = pd.read_csv(path)

    codes = [ "GOAL", "NEED", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "VALIDATION", "CONSTRAINTS", "SIDE_EFFECTS", "MATURITY_STAGE", ]

    # result = []
    # for repo in data['repo_id'].unique():
    #
    #     t = [repo,]
    #     for code in codes:
    #         t_data = data[(data['repo_id'] == repo) &
    #                       (data['final_annotation'].str.contains(code, na=False))].reset_index(drop=True)
    #
    #         t.append(len(t_data))
    #
    #     result.append(t)
    #
    # metrics_df = pd.DataFrame(result, columns=['repo_id'] + codes)
    # metrics_df.to_csv('dataset/per_repo_code.csv', index=False)
    # print(f"Metrics by repo and code saved to dataset/per_repo_code.csv.")
    #
    # result = []
    # for repo in data['repo_id'].unique():
    #
    #     t = [repo, ]
    #     for code in codes:
    #         t_data = data[(data['repo_id'] == repo) &
    #                       (data['final_annotation'].str.contains(code, na=False))].reset_index(drop=True)
    #
    #         t.append(len(t_data['commit_id'].unique()))
    #
    #     result.append(t)
    #
    # metrics_df = pd.DataFrame(result, columns=['repo_id'] + codes)
    # metrics_df.to_csv('dataset/per_repo_commit.csv', index=False)
    # print(f"Metrics by repo and commit saved to dataset/per_repo_commit.csv.")

    result = []
    for repo in data['repo_id'].unique():

        t = [repo, ]

        t_data = data[(data['repo_id'] == repo) & ~(data['final_annotation'] == "UNCODED")].reset_index(drop=True)

        t.append(len(t_data['commit_id'].unique()))

        result.append(t)

    metrics_df = pd.DataFrame(result, columns=['repo_id', 'commit_count'])
    metrics_df.to_csv('dataset/per_repo_commit_count.csv', index=False)
    print(f"Metrics by repo and commit count saved to dataset/per_repo_commit_count.csv.")


def calculate_metrics_by_cluster(path):
    data = pd.read_csv(path)

    codes = ["GOAL", "NEED", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "VALIDATION", "CONSTRAINTS", "SIDE_EFFECTS",
             "MATURITY_STAGE", ]

    # result = []
    # for cluster in data['cluster'].unique():
    #
    #     t = [cluster, ]
    #     for code in codes:
    #         t_data = data[(data['cluster'] == cluster) &
    #                       (data['final_annotation'].str.contains(code, na=False))].reset_index(drop=True)
    #
    #         t.append(len(t_data['commit_id'].unique()))
    #
    #     result.append(t)
    #
    # metrics_df = pd.DataFrame(result, columns=['cluster'] + codes)
    # metrics_df.to_csv('dataset/per_cluster_code.csv', index=False)
    # print(f"Metrics by cluster and code saved to dataset/per_cluster_code.csv.")

    result = []
    for cluster in data['cluster'].unique():
        t = [cluster, ]

        t_data = data[(data['cluster'] == cluster) & ~(data['final_annotation'] == "UNCODED")].reset_index(drop=True)

        t.append(len(t_data['commit_id'].unique()))

        result.append(t)

    metrics_df = pd.DataFrame(result, columns=['cluster', 'commit_count'])
    metrics_df.to_csv('dataset/per_cluster_commit_count.csv', index=False)
    print(f"Metrics by cluster and commit count saved to dataset/per_cluster_commit_count.csv.")


def voting_response(config, test=False):
    exp_ids = config['exp_ids']
    model_names = config['model_names']
    runs = 3

    for exp_id in exp_ids:
        for model_name in model_names:
            print(f"Processing experiment {exp_id} with model {model_name}...")

            input_path = f"dataset/prompts/component_identification{'/test' if test else ''}/all_{'test' if test else 'dev'}_data.csv"
            res_col = f'response_label_{exp_id}_{model_name}'
            data = pd.read_csv(input_path)

            res = pd.DataFrame([], columns=data.columns.tolist() + [res_col])
            for i, row in data.iterrows():
                res_code = []
                for code in content_identification_codes:
                    count = 0
                    for run in range(runs):
                        col = f'response_label_{exp_id}_{model_name}_{run}'
                        if code in row[col]:
                            count += 1
                    if count >= 2:
                        res_code.append(code)

                codes = ', '.join(res_code)

                t_row = row.copy()
                t_row[res_col] = codes

                res.loc[len(res)] = t_row

            # save the result to a new file
            output_path = f"dataset/prompts/component_identification{'/test' if test else ''}/{exp_id}/{model_name}/prompt_template_with_response_runs_combined.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            res.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}.")


if __name__ == "__main__":
    config = {
        "exp_ids" : [
            "1.0.0.0",
            "1.0.0.2",
            "1.1.0.2",
            # "1.2.0.2",
            "1.1.1.2" # Best Prompt Type in test
        ],
        "model_names" : [
            "o4-mini",
            # "gpt-5"
        ],
        "runs": 3
    }

    create_development_data(all_data, development_data, test_data)

    # # Development data
    base_prompt_design(config, development_data, base_prompt_results, test=False)
    get_response_from_prompt(config, base_prompt_results, prompt_response, test=False)
    
    save_response_to_development_data(config, development_data, test=False)
    voting_response(config, test=False)

    # # Test data
    # base_prompt_design(config, test_data, base_prompt_results, test=True)
    # get_response_from_prompt(config, base_prompt_results, prompt_response, test=True)
    
    # save_response_to_development_data(config, test_data, test=True)
    # voting_response(config, test=True)
