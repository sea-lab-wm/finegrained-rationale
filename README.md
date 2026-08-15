# Recovering Fine-Grained Code Change Rationale from Multiple Software Artifacts

This repository contains the replication package of "Recovering Fine-Grained Code Change Rationale from Multiple Software Artifacts". The content description of this repository can be found in ```Directory Structure``` section.

## Running *ARGUS*

![*ARGUS* Workflow](argus.png)

### Prerequisite
- Python 3.10
- OpenAI Api Key
- Github Access Token

### Environment Set-Up
1. Create a virtual environment: ```python -m venv venv```
2. Activate the environment: ```source venv/bin/activate```
3. Install required packages: ```pip install -r requirements.txt```
4. Add OpenAI API key as the environment variable: ```export OPENAI_TOKEN=<Your OpenAI Api Key>```
5. Add Github Access Token as the environment variable: ```export GITHUB_TOKEN=<Your Github Access Token>```

### Execution
Run the following commands sequentially from the root directory:
1. ```python scripts/a_DatasetPreprocessor.py```: Preprocess & sample commits from original dataset.
2. ```python scripts/b_ArtifactRetriever.py```: Retrieve information from artifacts for each commit. By default this script parse the information into sentences.
3. ```python scripts/c_RationaleComponentExtractor.py```: Produce the results of component identification(*CI*) for different prompts using developement data. To generate the results of *ARGUS*'s CI, run ```python scripts/c_RationaleComponentExtractor.py -t test```.
4. ```python scripts/d_RationaleComponentGenerator.py```: Produce the results of component generation(*CG*) for different prompts using developement data. To generate the results of *ARGUS*'s CG, run ```python scripts/d_RationaleComponentGenerator.py -t test```. The manual evaluation results can be found at ```results/CG/ManualEvaluationResult.csv```

## Directory Structure

### Data
- **data/AnnotatedSentenceData.csv**: Code change rationale dataset with manually annotated rationale components. (Referenced in *Section 1 - Introduction*)
- **data/AnnotationCodebook.csv**: The annotation codebook shared between the annotators.
- **data/CGPromptTemplate.csv**: All prompt templates for component generation.
- **data/CIPromptTemplate.csv**: All prompt templates for component identification.
- **data/FilteredCommits.csv**: Full list of commits after the applied exclusion criteria. (Referenced in *Section 3.1 - Commit Collection*)
- **data/sampled messages.csv**: Original dataset from Tian et al.(https://dl.acm.org/doi/10.1145/3510003.3510205)

### Scripts
- **scripts/utils/consts.py**: This script holds the common constants used for artifact retrival, component identification & component generation.
- **scripts/utils/functions.py**: This script holds the common functions used for artifact retrival, component identification & component generation.
- **scripts/a_DatasetPreprocessor.py**: This script filter commits and prepare the data for artifacts collection.
- **scripts/b_ArtifactRetriever.py**: This script collect artifacts sentences from different source for each commit.
- **scripts/c_RationaleComponentExtractor.py**: This script is used for prompt development and evaluation of component identionfication for *ARGUS*
- **scripts/d_RationaleComponentGenerator.py**: This script is used for prompt development and evaluation of component generation for *ARGUS*

### Results
- **results/AnnotatorAgreement.csv**: This csv shows the agreement rate and accuracy of the annotations in each annotation session. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/CommitWiseArtifactCount.csv**: This csv shows the number of Artifact we collected for each commit. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/ProjectWiseArtifactCount.csv**: This csv shows the number of Artifact we collected for each project. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/PromptDevelopmentPerformanceComparisonIndividualVSVoting.pdf**: This table shows performance comparison between voting and individual response of a model. (Referenced in *Section 5.3 - Prompt Development Results*)
- **results/CI/DevResult.csv**: This table shows the prompt development results of component identification task.
- **results/CI/EvaluationResult.csv**: This table shows all the results of *ARGUS* (only *GPT-o4-mini*) on evaluaion data in component identification task.
- **results/CI/CrossModelEvaluationResult.csv**: This table shows the results of *ARGUS* using 3 models including *GPT-5.2* and *Gemini-3-Flash* models on evaluaion data in component identification task. 
- **results/CG/DevResult.csv**: This table shows the manual dev results of generated rationale components used for *ARGUS*.
- **results/CG/EvaluationResult.csv**: This table shows all the manual evaluations of generated rationale components used for *ARGUS* including the evaluations of *GPT-5.2* and *Gemini-3-Flash*.
- **results/consistency/CrossPromptIdentificationConsistancy.csv**: 
- **results/RationaleGenerationSimilarity.csv**:

### User Study
- **user-study/UserStudyCommitDistribution.pdf**: This table shows the distribution of commits used in user study. (Referenced in *Section 6.1 - Methodology*)
- **user-study/FinalQuestionnaire.docx**: This table shows the distribution of commits used in user study. (Referenced in *Section 6.1 - Methodology*)
