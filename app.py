from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.form["message"]
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()


