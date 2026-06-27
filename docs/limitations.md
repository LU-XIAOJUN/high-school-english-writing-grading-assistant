# Limitations

This project is an educational NLP prototype for teacher assistance. It should not be described or used as a mature automatic essay scoring system.

## Model Dependence

The prototype relies on a local vision-language model through Ollama. OCR quality and feedback quality depend on the selected model, model version, local hardware, prompt length, handwriting quality, and image quality.

## No Custom Model Training

The project does not train a custom scoring model. It uses prompt-based OCR and rubric-based feedback generation through an existing local model.

## No Formal Evaluation Study

The repository does not include:

- a manually annotated benchmark dataset;
- accuracy metrics for OCR;
- scoring validity experiments;
- inter-rater reliability analysis;
- comparison against human raters;
- a controlled classroom deployment study.

## Human Review Required

All model-generated OCR text, scores, and feedback should be reviewed by a teacher. The application includes a human-in-the-loop review step for this reason.

## Educational Context

The current demonstration focuses on Chinese Gaokao continuation writing because it is a common and practical task in senior high school English teaching. The workflow can be adapted to other prompt-based writing tasks with different prompts and rubrics, but such adaptation would require further prompt design, validation, and review.

## Privacy Scope

The sample answer sheets included in this repository are anonymized or self-created demonstration samples. They do not contain student names, school names, class information, contact details, examination IDs, or other personally identifiable information. Users are responsible for ensuring that any additional data they process comply with applicable privacy, institutional, and ethical requirements.
