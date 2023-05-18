# 🎬 HSE Project Okko
Project in university on creating RecSys for Okko, mentored by <a href="https://github.com/kshurik" target="_blank">Shuhrat Khalilbekov</a> and <a href="https://github.com/5x12" target="_blank">Andrew Wolf</a>.
## ⛏️ Team members
* <a href="https://github.com/missukrof" target="_blank">Anastasiya Kuznetsova</a>
* <a href="https://github.com/aliarahimckulova" target="_blank">Aliya Rahimckulova</a>
* <a href="https://github.com/PBspacey" target="_blank">Nikita Senyatkin</a>
* Tigran Torosyan
## 🌙 Project report in Notion
- <a href="https://www.notion.so/Okko_project-4327569d110c4e949d042abcd310f1ae" target="_blank">Report</a> - report for project (under development);
# 🔗 Full RecSys Pipeline
Here we have the full pipeline to train and make inference using two-level model architecture.
<br>
<br>**Inspiration sources:**
* <a href="https://github.com/kshurik/rekkobook/tree/main/supplements/recsys" target="_blank">Project architecture</a>
* <a href="https://github.com/sharthZ23/your-second-recsys/blob/master/lecture_5/tutorial_hybrid_model.ipynb" target="_blank">Tutorial on a two-stage model</a>
* <a href="https://www.kaggle.com/code/sharthz23/implicit-lightfm/notebook" target="_blank">Implicit & LightFM</a>
* <a href="https://github.com/kshurik/rekkobook/blob/main/notebook_drafts/full_recsys_pipeline.ipynb" target="_blank">Full RecSys pipeline</a>
## 📁 Repo Structure
- <a href="https://github.com/missukrof/project-okko-final/tree/main/app" target="_blank">app</a> - application folder;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/artefacts" target="_blank">artefacts</a> - local storage for models artefacts;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/configs" target="_blank">configs</a> - local storage for the configuration files;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/data" target="_blank">data</a> - data local storage;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/data_prep" target="_blank">data_prep</a> - data preparation modules to be used during training_pipeline;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/draft_notebooks" target="_blank">draft_notebooks</a> - pipeline drafts in jupyter notebook format;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/models" target="_blank">models</a> - model fit and inference pipeline;
- <a href="https://github.com/missukrof/project-okko-final/tree/main/utils" target="_blank">utils</a> - some common functions that can be used everywhere.
## ‍💻 Basic files
- <a href="https://github.com/missukrof/project-okko-final/blob/main/preprocessing.py" target="_blank">preprocessing.py</a> - data preprocessing & feature engineering in one file;
- <a href="https://github.com/missukrof/project-okko-final/blob/main/train.py" target="_blank">train.py</a> - two-stage model training (the first level - LightFM, the second - CatBoost classifier);
- <a href="https://github.com/missukrof/project-okko-final/blob/main/inference.py" target="_blank">inference.py</a> - get recommendations from two-stage model for a particular user.
# Documentation
## Purpose
This project is a docker-packed ML solution for solving the problems of personal movie recommendation. The model consists of two levels: 
- The first level returns a list of the most relevant films based on content similarity and user preferences
- The second level ranks recommendations based on additional generated features
## Input
- <b>interactioins</b> - information about users interactions
- <b>movies_metadata</b> - movies metadata
both files should be in <i>.parquet</i> format.
## Pipeline
1. <b>Data gathering</b> - app gathers additional data to the movies using API
2. <b>Data processing and feature engineering</b> - based on processing gathered and input data, app creates new features for movies and also creates new dataframe "users_metadata.parquet" with generated user features
3. <b>Model training</b> - Lightfm model is trained on item/user similarities, then Catboost model is trained based on generated features
4. <b>Inference</b> - model predicts for a specific user
## Basic files to run
- <a href="https://github.com/missukrof/project-okko-final/blob/main/preprocessing.py" target="_blank">preprocessing.py</a> - use to preprocess data and generate features
- <a href="https://github.com/missukrof/project-okko-final/blob/main/train.py" target="_blank">train.py</a> - use to train both models
- <a href="https://github.com/missukrof/project-okko-final/blob/main/inference.py" target="_blank">inference.py</a> - use to get recommendations from two-stage model for a particular user.
## Output
table with movies list for specific user and calculated score and rank for each movie.
## How to run
1. "poetry export -f requirements.txt --output requirements.txt" - copy required poetry settings
2. - if mac: run "start.sh" 
   - if windows: "docker-compose up -d --build" 
App is running in localhost








