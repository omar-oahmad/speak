from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    dataUser = request.form['a']
    dataEmbed = request.form['b']
    dataPreference = request.form['c']
    arr = np.array([[dataUser, dataEmbed, dataPreference]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "_main_":
    app.run(debug=True)