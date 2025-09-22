import argparse
import multiprocessing as mp
import os.path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Optional
import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm
from tqdm.auto import tqdm

from utils.consts import *
from utils.functions import *


def get_code_information(template: str, codebook):
    result = []

    for i, row in codebook.iterrows():

        example_data[row['Annotation Labels']] = example_data.apply(
            lambda x: row['Annotation Labels'] in x['final_annotation'], axis=1)
        t_example_data = example_data[example_data[row['Annotation Labels']]]
        examples = t_example_data[['sentence']].drop_duplicates()

        filled = template
        replacements = {
            "<index>": str(i + 1),
            "<code>": row['Annotation Labels'],
            "<definintion>": row['Description'],
            "<question>": row['Component Expressed as Question'],
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
    try:
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
    except:
        print(commit_info['files'])

    result = "\n\n".join(result)
    return result


def get_sentence_information(template: str, commit_data):
    print(commit_data['commit_id'])

    result = {
        "COMMIT_MESSAGE": [],
        "CLASS_JAVADOCS": {},
        "METHOD_JAVADOCS": {},
        "CODE_COMMENT": [],
        "ISSUE": {},
        "PULL_REQUEST": {},
        "CODE_REVIEW_COMMENT": {}
    }

    for i, sentence in enumerate(commit_data['sentence']):
        sent = sentence['sentence']

        filled = template
        replacements = {
            "<index>": sentence['id'],
            "<source>": sentence['source'],
            "<sentence>": sent,
            "<labels>": sentence['annotations']
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
                source_placeholder = "the comments of the changed code. The sentences have [ADD] and [REM] prefix. The sentences with [ADD] means these sentences are added in code commit and sentences with [REM] means these sentences are removed in code commit"
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


def base_prompt_design(config, dev_path, output_path, test):
    print("\n\nStarting base prompt design...")

    experminet_ids = config['exp_ids']
    runs = config['runs']
    model_names = config['model_names']
    comp_iden_config_exp_id = config['comp_iden_config']['exp_id']
    comp_iden_config_model_name = config['comp_iden_config']['model_name']

    # gt_column = f"response_label_{comp_iden_config_exp_id}_{comp_iden_config_model_name}"
    gt_column = 'final_annotation'
    dev_path = f"data/prompts/component_identification{'/test' if test else ''}/{comp_iden_config_exp_id}/{comp_iden_config_model_name}/prompt_template_with_response_runs_combined.csv"

    data = pd.read_csv(dev_path)

    prompt_templates = pd.read_csv("data/CGPromptTemplate.csv")

    codebook = pd.read_csv('data/AnnotationCodebook.csv')
    codebook = codebook[codebook['Annotation Labels'].isin(codes)].reset_index(drop=True)

    data_json = []
    for index, commit_id in enumerate(data['commit_id'].unique()):
        if commit_id != 8083:
            continue
        temp_data = {}

        commit_df = data[(data['commit_id'] == commit_id)].copy().reset_index(drop=True)
        commit_df[gt_column] = commit_df[gt_column].fillna("UNCODED")
        commit_df['ground_truth'] = commit_df['ground_truth'].fillna("")

        gt_rows = commit_df[(commit_df['ground_truth'].notna())].reset_index(drop=True)

        first_row = commit_df.iloc[0]

        temp_data['commit_id'] = str(first_row['commit_id'])
        try:
            temp_data['rationale'] = gt_rows.iloc[0]['ground_truth']
        except Exception as e:
            # print wanrning
            print(f"Warning: No ground truth found for commit_id {commit_id}. Setting rationale to empty string.")
        temp_data['commit_url'] = first_row['commit_url']

        temp_data['sentence'] = []
        for sid, row in commit_df[commit_df[gt_column] != "UNCODED"].iterrows():
            if any(code in row[gt_column] for code in codes):
                if "_JAVADOCS" in row['source']:
                    class_part = row['sentence'].split(":")

                    class_name = class_part[0].strip().lstrip('[').rstrip(']')
                    javadocs = ":".join(class_part[1:]).strip()

                    temp_data['sentence'].append({
                        "id": row['id'],
                        "source": row['source'],
                        "class_name": class_name,
                        "sentence": javadocs,
                        "annotations": row[gt_column] if pd.notna(row[gt_column]) else "UNCODED",
                    })

                elif ("ISSUE" in row['source']
                      or "PULL_REQUEST" in row['source']
                      or "CODE_REVIEW_COMMENT" in row['source']):
                    issue_url_parts = row['source_url'].split('/')

                    temp_data['sentence'].append({
                        "id": row['id'],
                        "source": row['source'],
                        "issue_id": "/".join(issue_url_parts[:6] + [issue_url_parts[6].split("#")[0]]),
                        "sentence": row['sentence'],
                        "annotations": row[gt_column] if pd.notna(row[gt_column]) else "UNCODED",
                    })

                else:
                    temp_data['sentence'].append({
                        "id": row['id'],
                        "source": row['source'],
                        "sentence": row['sentence'],
                        "annotations": row[gt_column] if pd.notna(row[gt_column]) else "UNCODED",
                    })

        data_json.append(temp_data)

    for experiment_id in experminet_ids:
        prompt_records = []

        print("+" * 50)
        print(f"Experiment ID: {experiment_id}")
        experiment = experiment_id.split('.')

        example = 0 if experiment[1] == '0' else f"{experiment[1]}"

        template = prompt_templates[prompt_templates['version'] == experiment_id].iloc[0]

        print(f"Example: {example}")

        for i, commit in enumerate(data_json):
            code_diff_information = get_code_diff_information(template['code_diff_information'], commit)
            sentences_information = get_sentence_information(template['sentences_information'], commit)
            # actual_annotations = [s['annotations'] if s['annotations'] else "UNCODED" for s in commit['sentence']]

            prompt = template['template']
            prompt = prompt.replace("<project_name>", commit['commit_url'].split('/')[4])
            prompt = prompt.replace("<code_diff_information>", code_diff_information)
            prompt = prompt.replace("<sentences_information>", sentences_information)
            prompt = prompt.replace("<sentence_count>", str(len(sentences_information)))

            prompt_records.append({
                'example': example,
                'experiment_version': template['version'],
                'commit_id': commit['commit_id'],
                'commit_url': commit['commit_url'],
                'rationale': commit['rationale'],
                'prompt': prompt,
            })

        prompt_template_tosend = pd.DataFrame(prompt_records)

        t_output_path = output_path.replace("<exp_id>", f"{'test/' if test else ''}{experiment_id}")

        print(f"Saving prompt templates to {t_output_path}...")
        os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
        prompt_template_tosend.to_csv(t_output_path, index=False)


def get_prompt_response(prompt, model_name="o4-mini", temp=1):
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_TOKEN") or None)

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            reasoning={"effort": "high",}
        )

        resp = response.output_text  # direct text output in new API
        return resp.strip() if resp else ""
    except Exception as e:
        print(f"Error generating response for prompt: {e}")
        return ""

def get_gpt5_prompt_response(prompt, model_name="gpt-5-chat-latest", temp=0.1):
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_TOKEN") or None)

        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "user", "content": prompt}
            ],
            temperature=temp
        )

        resp = response.output_text  # direct text output in new API
        return resp.strip() if resp else ""
    except Exception as e:
        print(f"Error generating response for prompt: {e}")
        return ""


# def get_response_from_prompt(config, input_path, output_path, test=False):
#     exp_ids = config['exp_ids']
#     model_names = config['model_names']
#     runs = config['runs']
#
#     tqdm.pandas()
#     for exp_id in exp_ids:
#         for model_name in model_names:
#             for run in range(runs):
#                 try:
#                     print(f"Generating responses for experiment {exp_id} with model {model_name} ; Run {run}...")
#
#                     t_input_path = input_path.replace("<exp_id>", f"{'test/' if test else ''}{exp_id}")
#
#                     t_output_path = (output_path
#                                      .replace("<exp_id>", f"{'test/' if test else ''}{exp_id}")
#                                      .replace("<model>", model_name)
#                                      .replace("<run>", str(run)))
#
#                     data = pd.read_csv(t_input_path)
#
#                     print("\n\nStarting response generation from prompt...")
#
#                     data['response'] = data['prompt'].progress_apply(lambda x: get_prompt_response(x, model_name))
#
#                     data['input_token_count'] = data['prompt'].progress_apply(lambda x: get_token_count(model_name, x))
#                     data['output_token_count'] = data['response'].progress_apply(
#                         lambda x: get_token_count(model_name, x))
#
#                     # Save the results to the output path
#                     os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
#                     data.to_csv(t_output_path, index=False)
#                     print(f"Responses saved to {t_output_path}.")
#                 except Exception as e:
#                     print(f"Error processing experiment {exp_id} with model {model_name} ; Run {run}: {e}")
#                     continue


def _process_one(args):
    exp_id, model_name, run, input_path, output_path, test = args
    try:
        print(f"[PID {os.getpid()}] Generating responses for experiment {exp_id} "
              f"with model {model_name} ; Run {run}...")

        # Resolve paths
        prefix = 'test/' if test else ''
        t_input_path = input_path.replace("<exp_id>", f"{prefix}{exp_id}")
        t_output_path = (
            output_path
            .replace("<exp_id>", f"{prefix}{exp_id}")
            .replace("<model>", model_name)
            .replace("<run>", str(run))
        )

        tqdm.pandas(desc=f"Generating responses [model:{model_name}, exp:{exp_id}, run:{run}]")

        data = pd.read_csv(t_input_path)

        # per-row work with a progress bar
        data['response'] = data['prompt'].progress_apply(
            lambda x: get_prompt_response(x, model_name)
        )

        # Write output
        os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
        data.to_csv(t_output_path, index=False)

        msg = f"Responses saved to {t_output_path}."
        print(msg)
        return (exp_id, model_name, run, None, msg)

    except Exception as e:
        err = (f"Error processing experiment {exp_id} with model {model_name} ; Run {run}: {e}")
        print(err)
        return (exp_id, model_name, run, err, None)


def get_response_from_prompt(config, input_path, output_path, test=False, max_workers=None):
    exp_ids = config['exp_ids']
    model_names = config['model_names']
    runs = config['runs']

    # Build task list
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

    ctx = mp.get_context("spawn")

    print(f"Launching {len(tasks)} tasks with {max_workers} worker(s)...")

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
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                      precision_score + recall_score) > 0 else 0
        accuracy_score = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

        result['code_wise'][code] = [tp, fp, tn, fn, precision_score, recall_score, f1_score, accuracy_score]

    precision_score = tps / (tps + fps) if (tps + fps) > 0 else 0
    recall_score = tps / (tps + fns) if (tps + fns) > 0 else 0
    f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                  precision_score + recall_score) > 0 else 0
    accuracy_score = (tps + tns) / (tps + fps + tns + fns) if (tps + fps + tns + fns) > 0 else 0

    result['overall'] = [tps, fps, tns, fns, precision_score, recall_score, f1_score, accuracy_score]

    return result


def hermonic_mean(x, y, n=1):
    x = float(x)
    y = float(y)

    return ((1 + n * n) * x * y) / ((n * n) * x + y)


def evaluate_rationale(commit, model_name):
    eval_template_path = "dataset/prompts/prompt_templates_rationale_generation.csv"
    eval_template = pd.read_csv(eval_template_path)
    eval_template = eval_template[eval_template['version'] == "3.0"].iloc[0]['template']

    commit_url = commit['commit_url']
    project_name = commit_url.split('/')[4]

    eval_template = eval_template.replace("<project_name>", project_name)
    eval_template = eval_template.replace("<ground_truth_components>", commit.get('rationale', ""))
    eval_template = eval_template.replace("<generated_components>",
                                          commit['response'] if pd.notna(commit['response']) else "")

    eval_response = get_prompt_response(eval_template, model_name)
    return pd.Series({
        "eval_prompt": eval_template,
        "eval_response": eval_response,
    })


def parse_rationales(text):
    try:
        rationales = text.splitlines() if pd.notna(text) else []
    except Exception as e:
        pass
    res = []

    for i, rationale in enumerate(rationales):
        if rationale.strip():
            parts = rationale.split(':')
            code = parts[0].strip()
            description = ':'.join(parts[1:]).strip()

            if code in content_identification_codes:
                if len(description.split()) >= 3 and not ("Omitted".lower() in description.lower()):
                    res.append({'code': code, 'description': description})
                else:
                    if (description.strip().endswith(')')) or (i == (len(rationales) - 1)):
                        if "Fix test".lower() in description.lower() or "Correct checkstyle".lower() in description.lower():
                            res.append({'code': code, 'description': description})
                            continue
                        print(f"Skipping rationale with code {code} due to insufficient description length.")
                        print(f"Rationale: {description}")
                    else:
                        res.append({'code': code, 'description': description})
            else:
                print(f"Skipping rationale with code {code} as it is not in the content identification codes.")
                print(f"Rationale: {description}")

    return res


def metrics_calculation_response(row):
    result = []
    pred_rationale = row['response']
    actual_rationale = row['rationale']
    eval_response = row['evaluation']

    pred_rationales = parse_rationales(pred_rationale)
    actual_rationales = parse_rationales(actual_rationale)

    # calculate tp, fp, tn , fn

    pred_codes = [r['code'] for r in pred_rationales]
    actual_codes = [r['code'] for r in actual_rationales]

    comp_wise_confusion_matrix = {}

    total_tp = total_fp = total_tn = total_fn = 0

    for code in content_identification_codes:
        tp = fp = tn = fn = 0
        if code in pred_codes and code in actual_codes:
            tp += 1
        if code in pred_codes and code not in actual_codes:
            fp += 1
        if code not in pred_codes and code in actual_codes:
            fn += 1
        if code not in pred_codes and code not in actual_codes:
            tn += 1

        comp_wise_confusion_matrix[code] = {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

    try:
        data = json.loads(eval_response)
    except Exception as e:
        data = json.loads(eval_response.replace('“', '"').replace('”', '"'))

    max_val = 5

    q1s_yes = []
    q1s_no = []
    q2s = []
    q3s = []
    hms = []
    q2_focus_hms = []

    comp_wise_q1_yes = {}
    comp_wise_q1_no = {}
    comp_wise_q2 = {}
    comp_wise_q3 = {}
    component_wise_hms = {}
    component_wise_q2_focus_hms = {}
    for key in data:
        if key['Q1-Answer'] == "YES":
            if key['rationale_component'] not in content_identification_codes:
                continue

            q1s_yes.append(1)
            q2 = int(key['Q2-Answer'])
            q3 = int(key['Q3-Answer'])
            q3 = max_val - q3 + 1

            q2s.append(q2)
            q3s.append(q3)

            t_hm = hermonic_mean(q2, q3)
            t_q2_focus_hms = hermonic_mean(q3, q2, 2)

            hms.append(t_hm)
            q2_focus_hms.append(t_q2_focus_hms)

            comp_wise_q1_yes[key['rationale_component']] = 1
            comp_wise_q2[key['rationale_component']] = q2
            comp_wise_q3[key['rationale_component']] = q3

            component_wise_hms[key['rationale_component']] = t_hm
            component_wise_q2_focus_hms[key['rationale_component']] = t_q2_focus_hms
        else:
            q1s_no.append(1)
            comp_wise_q1_no[key['rationale_component']] = 1

    for code in content_identification_codes:
        result.append({
            "commit_id": row['commit_id'],
            'component': code,
            'tp': comp_wise_confusion_matrix[code]['tp'],
            'fp': comp_wise_confusion_matrix[code]['fp'],
            'tn': comp_wise_confusion_matrix[code]['tn'],
            'fn': comp_wise_confusion_matrix[code]['fn'],
            'q1s_yes': comp_wise_q1_yes.get(code, 0),
            'q1s_no': comp_wise_q1_no.get(code, 0),
            'q2s': comp_wise_q2.get(code, 0),
            'q3s': comp_wise_q3.get(code, 0),
            'hms': component_wise_hms.get(code, 0),
            "q2_focus_hms": component_wise_q2_focus_hms.get(code, 0)
        })

    result.append({
        "commit_id": row['commit_id'],
        'component': 'Total',
        'tp': total_tp,
        'fp': total_fp,
        'tn': total_tn,
        'fn': total_fn,
        'q1s_yes': sum(q1s_yes),
        'q1s_no': sum(q1s_no),
        'q2s': sum(q2s) / len(q2s) if len(q2s) > 0 else 0,
        'q3s': sum(q3s) / len(q3s) if len(q3s) > 0 else 0,
        'hms': sum(hms) / len(hms) if len(hms) > 0 else 0,
        "q2_focus_hms": sum(q2_focus_hms) / len(q2_focus_hms) if len(q2_focus_hms) > 0 else 0
    })

    t = pd.DataFrame(result, columns=pd.Series(
        ["commit_id", "component", "tp", "fp", "tn", "fn", "q1s_yes", "q1s_no", "q2s", "q3s", "hms", "q2_focus_hms"]))
    return t


def compute_metrics(config, input_path):
    exp_ids = config['exp_ids']
    model_names = config['model_names']
    comp_iden_config_exp_id = config['comp_iden_config']['exp_id']
    comp_iden_config_model_name = config['comp_iden_config']['model_name']
    runs = config['runs']
    eval_runs = config['eval_runs']

    all_result = []
    all_results_path = "dataset/prompts/rationale_generation/all_result.csv"

    for exp_id in exp_ids:
        experiment = exp_id.split('.')

        example = 0 if experiment[1] == '0' else f"{experiment[1]}"
        reference = "GT_SENTENCE" if experiment[-1] == '1' else "IDENTIFIED_SENTENCE"

        for model_name in model_names:
            for run in range(runs):
                for eval_run in range(eval_runs):
                    print(f"Processing experiment {exp_id}, model {model_name}, run {run}, eval run {eval_run}...")

                    t_input_path = (input_path
                                    .replace("<exp_id>", exp_id)
                                    .replace("<model_name>", model_name)
                                    .replace("<run>", str(run))
                                    .replace("<eval_run>", str(eval_run)))

                    data = pd.read_csv(t_input_path)

                    res = pd.DataFrame()
                    for i, commit in data.iterrows():
                        t = metrics_calculation_response(commit)

                        res = pd.concat([res, t])

                    for code in content_identification_codes + ['Total']:
                        t = res[res['component'] == code].reset_index(drop=True)

                        tp = t['tp'].sum()
                        fp = t['fp'].sum()
                        tn = t['tn'].sum()
                        fn = t['fn'].sum()
                        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                                      precision_score + recall_score) > 0 else 0
                        accuracy_score = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
                        q1s_yes = t['q1s_yes'].sum()
                        q1s_no = t['q1s_no'].sum()

                        avg_q2 = t['q2s'].mean()
                        avg_q3 = t['q3s'].mean()
                        avg_q2_q3 = (t['q2s'] + t['q3s']).mean()
                        avg_hm = t['hms'].mean()
                        q2_focus_hms = t['q2_focus_hms'].mean()

                        all_result.append({
                            'experiment_version': exp_id,
                            'model_name': model_name,
                            'run': run,
                            'eval_run': eval_run,
                            'example': example,
                            'reference': reference,
                            'component': code,
                            "tp": tp,
                            "fp": fp,
                            "tn": tn,
                            "fn": fn,
                            "precision": precision_score,
                            "recall": recall_score,
                            "f1_score": f1_score,
                            "accuracy": accuracy_score,
                            "q1s_yes": q1s_yes,
                            "q1s_no": q1s_no,
                            "avg_q2_q3": avg_q2_q3,
                            "q2_focus_hms": q2_focus_hms,
                            "avg_q2": avg_q2,
                            "avg_q3": avg_q3,
                            "avg_hm": avg_hm
                        })

    # save the results to the output path
    all_results_df = pd.DataFrame(all_result, columns=pd.Series([
        'experiment_version',
        'model_name',
        'run',
        'eval_run',
        'example',
        'reference',
        'component',
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "q1s_yes",
        "q1s_no",
        "avg_q2_q3",
        "q2_focus_hms",
        "avg_q2",
        "avg_q3",
        "avg_hm"
    ]))
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)
    all_results_df.to_csv(all_results_path, index=False)
    print(f"All results saved to {all_results_path}.")


def create_development_data(input_path, dev_path, example_path):
    data = pd.read_csv(input_path)
    example_data = [756, 8083]
    data['id'] = data.apply(lambda row: f"{row['commit_id']}_{row['source_id']}_{row['text_id']}_{row['sentence_id']}",
                            axis=1)

    print("\n\nCreating development data...")

    target_development_commit = 11
    development_data = pd.DataFrame([], columns=data.columns)

    repo = 5
    cluster = 4

    while True:
        if len(development_data['commit_id'].unique()) >= target_development_commit:
            break

        filtered_data = data[(data['repo_id'] == repo) & (data['cluster'] == cluster) & ~(
            data['commit_id'].isin(example_data))].reset_index(drop=True)
        filtered_commits = filtered_data[['commit_id']].drop_duplicates().reset_index(drop=True)

        sample_commit = filtered_commits.sample(n=1, random_state=42)
        sampled_df = filtered_data[filtered_data['commit_id'] == sample_commit['commit_id'].iloc[0]].reset_index(
            drop=True)

        development_data = pd.concat([development_data, sampled_df], ignore_index=True)

        repo = repo % 5 + 1
        cluster = cluster % 4 + 1

    # save the development data
    os.makedirs(os.path.dirname(dev_path), exist_ok=True)
    development_data.to_csv(dev_path, index=False)
    print(f"Development data saved to {dev_path}.")

    print("\n\nCreating Example data...")

    example_data = data[(data['commit_id'].isin(example_data))].reset_index(drop=True)

    # save the example data
    os.makedirs(os.path.dirname(example_path), exist_ok=True)
    example_data.to_csv(example_path, index=False)
    print(f"Example data saved to {example_path}.")


def calculate_metrice_by_component(input_path, output_path):
    target_experiment = ["1.0.0.0", "1.0.1.0", "1.0.1.1", "1.1.0.0", "1.1.1.0", "1.1.1.1", ]
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
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (
                                                                                                      precision_score + recall_score) > 0 else 0

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

    target_components = ["GOAL", "NEED", "CONSTRAINTS", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "VALIDATION",
                         "MATURITY_STAGE", "SIDE_EFFECTS", "UNCODED", ]
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
            t_data = data[
                (data['source'] == source) & (data['final_annotation'].str.contains(code, na=False))].reset_index(
                drop=True)
            t.append(t_data['commit_id'].nunique())

        result.append(t)

    metrics_df = pd.DataFrame(result[1:], columns=result[0])

    metrics_df.to_csv('dataset/per_source_code.csv', index=False)
    print(f"Metrics by source and component saved to dataset/per_source_code.csv.")


def calculated_metrics_by_repo(path):
    data = pd.read_csv(path)

    codes = ["GOAL", "NEED", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "VALIDATION", "CONSTRAINTS", "SIDE_EFFECTS",
             "MATURITY_STAGE", ]

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


# def get_eval_prompt(config, input_path, output_path):
#     exp_ids = config['exp_ids']
#     model_names = config['model_names']
#     runs = config['runs']
#
#     tqdm.pandas()
#
#     for exp_id in exp_ids:
#         for model_name in model_names:
#             for run in range(runs):
#                 print(f"Processing experiment {exp_id}, model {model_name}, run {run}...")
#
#                 t_input_path = input_path.replace('<exp_id>', exp_id).replace('<model_name>', model_name).replace('<run>', str(run))
#                 t_output_path = output_path.replace('<exp_id>', exp_id).replace('<model_name>', model_name).replace('<run>', str(run))
#
#                 data = pd.read_csv(t_input_path)
#
#                 t = data.progress_apply(lambda x: evaluate_rationale(x, model_name), axis=1)
#
#                 data['eval_prompt'] = t['eval_prompt']
#                 data['evaluation'] = t['eval_response']
#
#                 os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
#                 data.to_csv(t_output_path, index=False)
#                 print(f"Responses with evaluation saved to {t_output_path}.")


def _process_one_eval(args: Tuple[str, str, int, str, str]) -> Tuple[str, str, int, Optional[str], Optional[str]]:
    exp_id, model_name, run, eval_run, input_path, output_path = args
    try:
        print(f"[PID {os.getpid()}] Processing experiment {exp_id}, model {model_name}, run {run}...")

        # Resolve paths
        t_input_path = (
            input_path
            .replace('<exp_id>', exp_id)
            .replace('<model_name>', model_name)
            .replace('<run>', str(run))
        )
        t_output_path = (
            output_path
            .replace('<exp_id>', exp_id)
            .replace('<model_name>', model_name)
            .replace('<run>', str(run))
            .replace('<eval_run>', str(eval_run))
        )

        # Read
        data = pd.read_csv(t_input_path)

        # Apply evaluation per row (sequential inside this worker)
        # Expect evaluate_rationale(row, model_name) -> dict/Series with keys: eval_prompt, eval_response
        out = data.apply(lambda row: evaluate_rationale(row, model_name), axis=1)

        # Attach results
        # Keep output column names consistent with your original code
        data['eval_prompt'] = out['eval_prompt']
        data['evaluation'] = out['eval_response']

        # Write
        os.makedirs(os.path.dirname(t_output_path), exist_ok=True)
        data.to_csv(t_output_path, index=False)

        msg = f"Responses with evaluation saved to {t_output_path}."
        print(msg)
        return (exp_id, model_name, run, None, msg)

    except Exception as e:
        err = f"Error processing {exp_id}, {model_name}, run {run}: {e}"
        print(err)
        return (exp_id, model_name, run, err, None)


def get_eval_prompt(config, input_path, output_path, max_workers=None):
    exp_ids = config['exp_ids']
    model_names = config['model_names']
    runs = config['runs']
    eval_runs = config.get('eval_runs', 3)

    # Build tasks
    tasks = [
        (exp_id, model_name, run, eval_run, input_path, output_path)
        for exp_id in exp_ids
        for model_name in model_names
        for run in range(runs)
        for eval_run in range(eval_runs)
    ]

    if not tasks:
        print("No tasks to run.")
        return

    # Decide pool size
    if max_workers is None:
        max_workers = max(1, min(mp.cpu_count(), len(tasks)))

    ctx = mp.get_context("spawn")

    print(f"Launching {len(tasks)} tasks with {max_workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_process_one_eval, t) for t in tasks]
        for fut in as_completed(futures):
            exp_id, model_name, run, err, msg = fut.result()
            if err:
                # already printed inside worker; reiterate here if you want
                pass
            else:
                # already printed inside worker; keep quiet or log
                pass

    print("All tasks finished.")


def get_response():
    print("Type your prompt (or 'exit' / 'quit' / 'q' to stop).")
    while True:
        try:
            # prompt = input("Prompt: ").strip()
            prompt = ''

            if prompt.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break

            print("\n" + "=" * 50 + "\n")
            print("Getting responses... for gpt-5\n", flush=True)
            gpt_5_response = get_gpt5_prompt_response(prompt)
            print("GPT-5 Response:")
            print(gpt_5_response)
            print("\n" + "=" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n(Interrupted) Type 'exit' to quit, or press Enter to continue.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # add argparse: -t test for test data
    parser = argparse.ArgumentParser(description="Rationale Component Generator")
    parser.add_argument('-t', default="dev", choices=["dev", "test"], type=str, help="Use dev/test data")
    args = parser.parse_args()

    config = {
        "exp_ids": [
            "2.0.0",
            # "2.1.0",
            "2.2.0",
            # "2.0.1",
            # "2.1.1",
            # "2.2.1",
        ],
        "model_names": [
            # "gpt-4.1",
            "o4-mini",
        ],
        "runs": 3,
        "comp_iden_config": {
            "exp_id": "1.1.1.2",
            "model_name": "o4-mini"
        },
        "eval_runs": 3
    }

    if args.t == "test":
        base_prompt_design(config, test_data, base_prompt_rationale_geenration, test=True)
        get_response_from_prompt(config, base_prompt_rationale_geenration, prompt_response_rationale_generation, test=True)
        # compute_metrics(config,prompt_response_rationale_generation, test=True)
    else:
        base_prompt_design(config, development_data, base_prompt_rationale_geenration, test=False)
        get_response_from_prompt(config, base_prompt_rationale_geenration, prompt_response_rationale_generation, test=False)
        # compute_metrics(config,prompt_response_rationale_generation, test=False)

    # get_response()
