# Disaster Response Pipeline Project

### Project Description

This project is a part of Udacity Data Scientist Nanodegree Program. In this project, I have dealt with data set containing real messages that were sent during disastder events, I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. I built an ETL pipeline that cleaned messages by used NLP techniques. The text data was trained on a multioutput classifier model using random forest. finally I used Flask app that classifies input messages and shows visualizations of key statistics of the dataset.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the  app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



