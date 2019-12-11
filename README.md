# Disaster Response Pipeline Project

### Project Description

This project is a part of Udacity Data Scientist Nanodegree Program. In this project, I have dealt with data set containing real messages that were sent during disastder events, I created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. I built an ETL pipeline that cleaned messages by used NLP techniques. The text data was trained on a multioutput classifier model using random forest. finally I used Flask app that classifies input messages and shows visualizations of key statistics of the dataset.

### Data:

There are two dataset:
1- messages 2- categories. 
The data has provided by Figure Eight which is content tweets and text messages from real-life disasters, the disaster messages come from different communications, so I have to combine these two datasets and re-labels them.

### Featureset Exploration:
##### categories:
- id 
- categories

##### messages:
- id
- message
- original
- genre

### Software and Libraries:

- Python
- NumPy
- pandas
- scikit-learn 
- nltk
- plotly
- sqlalchemy 
- pickle
- re
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the  app's directory to run your web app.
    `python run.py`

3. open another Terminal Window and Type:

   `env|grep WORK`
   
   Then in a new web browser window, type in the following:
   
   `https://SPACEID-3001.SPACEDOMAIN
`




