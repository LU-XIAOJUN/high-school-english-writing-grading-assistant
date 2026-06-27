# Privacy Note

This repository contains a demonstration version of **High School English Writing Grading Assistant（高中英语作文批改助手）**.

## Sample Data

The sample answer sheets included in this repository are anonymized or self-created demonstration samples. They do not contain student names, school names, class information, contact details, examination IDs, or other personally identifiable information. They are provided only to demonstrate the application workflow.

The identifiers shown in the sample output, such as `S001`, `S002`, and `S003`, are demonstration labels only.

## Local Processing

The prototype is designed to run locally. It calls a local Ollama service for OCR and feedback generation. The application does not intentionally send answer-sheet images or recognized writing to a cloud API.

Users who modify the code or connect it to remote APIs are responsible for checking their own data-protection requirements.

## Uploading New Data

Before using this project with new answer sheets, users should confirm that they have permission to process and store the data. For public repositories or public demonstrations, use anonymized or self-created demonstration samples only.

## Logs and Temporary Files

Runtime logs may contain excerpts of model outputs or error messages. The `.gitignore` file excludes runtime logs by default. Do not upload unreviewed runtime logs if they contain sensitive information.

The `demo_snapshot/logs/` directory is intentionally kept for snapshot structure only. Public demo logs should remain empty or contain only reviewed, non-sensitive placeholder files.
