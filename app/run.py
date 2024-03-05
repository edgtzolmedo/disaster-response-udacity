import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# download necessary NLTK data
import sklearn.externals

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter, Pie, Treemap
import joblib
from sqlalchemy import create_engine

#  
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    
    # Plotting of Categories Distribution in Direct Genre
    direct_cate = df[df.genre == 'direct']
    # Convert the DataFrame to numeric, ignoring non-numeric values
    direct_cate = direct_cate.apply(pd.to_numeric, errors='coerce')
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    marker=dict(colors=plotly.colors.sequential.Rainbow),
                    hole=.3
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
            }
        },
        # category plotting (Visualization#2)
        {
            'data': [
                Treemap(
                    labels=cate_names,
                    parents=[""]*len(cate_names),
                    values=cate_counts,
                    marker=dict(colors=plotly.colors.sequential.Rainbow),
                    branchvalues="total"
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'autosize': False,
                'width': 800,  # Adjust as needed
                'height': 500  # Adjust as needed
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()