from flask import Flask, render_template, request, redirect
from pred_pipeline import preprocessing, vectorizer, get_prediction



app = Flask(__name__)


# Define global variables here
reviews = []
positive = 0
negative = 0




@app.route("/")
def index():
    data = {
        'reviews': reviews,
        'positive': positive,
        'negative': negative
    }
                    
    return render_template('index.html', data=data)   #pass data to index.html


@app.route("/", methods=['POST'])
def post():
    global positive, negative, reviews  #Tell Python to use global variables

    text = request.form['text']
    preprocessed_txt = preprocessing(text)
    vectorized_txt = vectorizer(preprocessed_txt,tokens)
    prediction = get_prediction(vectorized_txt)
    

    if prediction == 'negative':
        global negative
        negative += 1
    else:
        global positive
        positive += 1

    reviews.insert(0, text)

    return redirect("/") # redirect home page after processing home data


if __name__ == "__main__":
    app.run(debug=True)
