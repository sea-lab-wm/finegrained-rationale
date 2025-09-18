# Fine-grained Multi-Document Extraction and Generation of Code Change Rationale

This repository contains the replication package of "Fine-grained Multi-Document Extraction and Generation of Code Change Rationale". The contents of this repository can be found in [Directory](#directory) section.

## Directory

### Data
- **data/AnnotatedSentenceData.csv**: Code change rationale dataset with manually annotated rationale components. (Referenced in *Section 1 - Introduction*)
- **data/FilteredCommits.csv**: Full list of commits(830 commits) after the applied exclusion criteria. (Referenced in *Section 3.1 - Commit Collection*)
### Scripts
- **scripts/a_DatasetPreprocessor.py**: This script filter commits and prepare the data for artifacts collection.
- **scripts/b_ArtifactRetriever.py**: This script collect artifacts sentences from different source for each commit.
- **scripts/c_RationaleComponentExtractor.py**: This script is used for prompt development and evaluation of component identionfication for *ARGUS*
- **scripts/d_RationaleComponentGenerator.py**: This script is used for prompt development and evaluation of component generation for *ARGUS*
### Results
- **results/ProjectWiseArtifactCount.csv**: This csv shows the number of Artifact we collected for each project. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/CommitWiseArtifactCount.csv**: This csv shows the number of Artifact we collected for each commit. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/AnnotatorAgreement.csv**: This csv shows the agreement rate and accuracy of the annotations in each annotation session. (Referenced in *Section 3.4 - Results and Analysis*)
- **results/PromptDevelopmentPerformanceComparisonIndividualVSVoting.pdf**: This table shows performance comparison between voting and individual response of a model. (Referenced in *Section 5.3 - Prompt Development Results*)
- **results/UserStudyCommitDistribution.pdf**: This table shows the distribution of commits used in user study. (Referenced in *Section 6.1 - Methodology*)