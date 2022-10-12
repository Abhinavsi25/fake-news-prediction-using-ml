from turtle import title
from flask import Flask, render_template, request,redirect,url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


# app = Flask(__name__)
# vectorizer=pickle.load(open("vectorizer.pkl",'rb'))
# model = pickle.load(open('FakeNews.pkl', 'rb'))
# dataframe = pd.read_csv('train.csv')
# x = dataframe['text']
# y = dataframe['label']
# vectorizer = TfidfVectorizer()

# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state=20)

# def fake_news_det(news):
#     model = LogisticRegression()
#     X_test_prediction = model.predict(x)
#     input_data = [news]
#     vectorized_input_data = vectorizer.transform(input_data)
#     prediction = model.predict(vectorized_input_data)
#     return prediction

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         message = str(request.form['news'])
#         pred = fake_news_det(message)
#         print(pred)
#         return render_template('prediction.html',prediction_text="News headline is -> {}".format(pred) )
#     else:
#         return render_template('prediction.html')

# if __name__ == '__main__':
#     app.run(debug=True)


app = Flask(__name__)

# vectorizer=pickle.load(open("vectorizer.pkl",'rb'))
model=pickle.load(open("FakeNews.pkl",'rb'))
vectorizer = TfidfVectorizer()
app = Flask(__name__)


@app.route('/' )
def home():
    return render_template('indext.html')
     
        


@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == "POST":
        news=str(request.form["news"])
        print(news)
        predict=model.predict(vectorizer.transform[news])
        print(predict)
        
        return render_template("prediction.html",prediction_text="News headline is -> {}".format(predict))


if __name__=='__main__':
    app.debug=True
    app.run()