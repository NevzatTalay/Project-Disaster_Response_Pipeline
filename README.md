# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

## Table of Contents
1. [Project Definition](#description)
2. [What to Do to Run](#getting_started)
	1. [Required Libraries](#dependencies)
	2. [Installation and Running Scripts](#installation)
 	3. [Website and Graphs](#output)
3. [License](#license)
4. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Project Definition
The algorithm is trained on a dataset of over 50,000 disaster response messages. It is able to classify messages into 36 categories, including needs, safety, aid etc.

The Flask application makes it easy to use the algorithm. Simply type messages to the application, and it will generate a classification report. The report includes the category for each message.

Disaster Response Pipeline is a valuable tool for emergency responders and humanitarian organizations. It can help to quickly and efficiently identify the needs of people affected by a disaster.

<a name="getting_started"></a>
## What to Do to Run

<a name="dependencies"></a>
### Required Libraries

* [sys — System-specific parameters and functions](https://docs.python.org/3/library/sys.html)
* [re — Regular expression operations](https://docs.python.org/3/library/re.html)
* [Numpy](https://numpy.org/install/)
* [Pandas](https://pandas.pydata.org/)
* [nltk - Natural Language Toolkit](https://www.nltk.org/)
* [scikit-learn - Machine Learning in Python](https://scikit-learn.org/)
* [SQLAlchemy - the Python SQL toolkit](https://www.sqlalchemy.org/)
* [pickle — Python object serialization](https://docs.python.org/3/library/pickle.html)
* [Flask](https://flask.palletsprojects.com/en/3.0.x/)
* [Plotly](https://plotly.com/)

```
import re                                                                                    
import numpy as np                                                                           
import pandas as pd                                                                         
from nltk.tokenize import word_tokenize                                                      
from nltk.stem import WordNetLemmatizer                                                  
import nltk                                                                           
import pickle                                                                       
from sklearn.model_selection import GridSearchCV                                        
from sklearn.ensemble import RandomForestClassifier                                      
from sklearn.model_selection import train_test_split                                   
from sklearn.pipeline import Pipeline, FeatureUnion                                  
from sklearn.base import BaseEstimator, TransformerMixin                              
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer          
from sklearn.metrics import classification_report                                         
from sqlalchemy import create_engine                                                     
from sklearn.multioutput import MultiOutputClassifier                                    
from sklearn.neighbors import KNeighborsClassifier                                    
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer
```

<a name="installation"></a>
### Installation
To install this project, follow these steps:

1. Clone the repository:
git clone https://github.com/NevzatTalay/Project-Disaster_Response_Pipeline.git
2. Install dependencies:
cd Project-Disaster_Response_Pipeline
3. Run ETL Pipeline to get db file.
```
python process_data.py disaster_messages.csv disaster_categories.csv MessageDatabase.db
```
4. Run ML Pipeline to obtain pickle file.
```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```
5. Run Flask Application and open in browser.
```
cd app
python run.py
```
Open your browser and type https://0.0.0.0:3000/

<a name="license"></a>
<a name="output"></a>
### Website and Graphs
![alt text for screen readers](/resources/webpage view.png "General View of Web-Site")



## License
MIT License

Copyright (c) 2023 Nevzat Anıl Talay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

<a name="acknowledgement"></a>
## Acknowledgement
I would like to acknowledge the following parties for their contributions to this work:

* Figure-8, for making this dataset available to Udacity for training purposes.
* Udacity, for providing the training.
Please feel free to use the contents of this work, but please cite me, Udacity, and/or Figure-8 accordingly.
