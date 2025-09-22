codes = ["GOAL", "NEED", "ALTERNATIVES", "SELECTED_ALTERNATIVE", "VALIDATION", "SIDE_EFFECTS", "MATURITY_STAGE", "UNCODED"]
content_identification_codes = ["GOAL", "NEED", "ALTERNATIVES"]
sources = ["COMMIT_MESSAGE", "CLASS_JAVADOCS", "METHOD_JAVADOCS", "CODE_COMMENT", "ISSUE", "PULL_REQUEST", "CODE_REVIEW_COMMENT"]

### Preprocess Dataset - Tian et al. ###
dataset_path = "data/sampled messages.csv"
non_atomic_commits_path = "data/generated/non_atomic_commits.csv"
pop_data_with_commit_info = "data/generated/dataset_with_commit_info.csv"
pop_filtered_data = "data/generated/pop_filtered_data.csv"
sample_marked_data = "data/FilteredCommits.csv"
issue_pr_id_from_commit_msg = "data/issue_pr_id_from_commit_msg.csv"
preprocess_data = "data/generated/preprocess_data.json"

### Retrival Module ###
dataset_with_commit_info = "data/dataset_with_commit_info.json"
dataset_with_commit_issue_info = "data/dataset_with_commit_issue_info.json"
dataset_with_commit_issue_pr_info = "data/dataset_with_commit_issue_pr_info.json"
dataset_with_commit_issue_pr_cr_info = "data/dataset_with_commit_issue_pr_cr_info.json"
dataset_with_commit_issue_pr_cr_javadoc_info = "data/dataset_with_commit_issue_pr_cr_javadoc_info.json"
dataset_with_commit_issue_pr_cr_javadoc_ref_info = "data/dataset_with_commit_issue_pr_cr_javadoc_ref_info.json"
dataset_with_commit_issue_pr_cr_javadoc_add_ref_info = "data/dataset_with_commit_issue_pr_cr_javadoc_add_ref_info.json"

final_data_json = "data/final_data.json"
final_data_csv = "data/final_data.csv"

### Postprocess ###
dataset_sentence = "data/dataset_with_parsed_sentence.json"
dataset_filtered = "data/dataset_filtered.json"

### Prompt Design ###
all_data = "data/all_data.csv"
all_data_annotation_separated = "data/all_data_annotation_separated.csv"
development_data = "data/development_data.csv"
example_data = "data/example_data.csv"
test_data = "data/test_data.csv"
base_prompt_results = 'data/prompts/component_identification/<exp_id>/prompt_template_wo_response.csv'
base_prompt_rationale_geenration = 'data/prompts/rationale_generation/<exp_id>/prompt_template_wo_response.csv'
prompt_response = 'data/prompts/component_identification/<exp_id>/<model>/<run>/prompt_template_with_response.csv'
prompt_response_rationale_generation = 'data/prompts/rationale_generation/<exp_id>/<model>/<run>/prompt_template_with_response.csv'
promp_metrics = 'data/<exp_id>/dev_response.csv'
exp_component_metrics = 'data/exp_component_metrics.csv'
prompt_response_result = 'data/prompt_template_with_response_result.csv'

### Util ###
annotation_data = "source/prompt_engineering/data/annotation_data.csv"
development_data = "source/prompt_engineering/data/development_data.csv"

### Preloaded Data Process ###
preloaded_commits = "data/preload/TrainingData.csv"
commit_msg_process = "data/TrainingData_commit_msg_process.csv"
add_additional_reference = "data/TrainingData_additional_ref.csv"
dataset_duplicate_ref_removed = "data/TrainingDataDuplicateRefRemoved.csv"
dataset_cr_info_added = "data/TrainingDataCrInfoAdded.csv"
dataset_issue_info_added = "data/TrainingDataIssueInfoAdded.csv"
dataset_pr_review_info_added = "data/TrainingDataPRReviewInfoAdded.csv"
dataset_reference_info_added = "data/TrainingDataRefInfoAdded.csv"
dataset_cluster = "data/preload/TrainingDataCluster.csv"
dataset_formatted = "data/TrainingDataFormatted.csv"
annotation_data = "data/annotation_data.csv"
pilot_dataset = "data/pilot_dataset.csv"
dataset_formatted_sent = "data/TrainingDataFormattedSent.csv"
sample_dataset = "data/sample_dataset.csv"
final_annotation_dataset_parsed = "data/final_annotation_dataset_parsed.csv"
annotation_agreement_analysis = "results/annotation_agreement_analysis.csv"
final_annotation_dataset = "data/annotation_data.csv"