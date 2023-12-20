from flask import Flask, request, render_template
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/test_flask'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the trained model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("Model_DatasetLengkap")
tokenizer = AutoTokenizer.from_pretrained("Model_DatasetLengkap")

def translate_sentence(sentence):
    inputs = tokenizer([sentence], return_tensors="pt")
    outputs = model.generate(**inputs)
    translated_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_sentence

def compute_bleu_score(reference, translation):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], translation, smoothing_function=smoothie)
     

# Define a simple model
class History(db.Model):
    _tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    slang_sentence = db.Column(db.String(256), nullable=True)
    reference_sentence = db.Column(db.String(256), nullable=True)
    translate_sentence = db.Column(db.String(256), nullable=True)
    bleu_score = db.Column(db.Float(10,10), nullable=True)

    # Define a simple model
class Data(db.Model):
    _tablename__ = 'data'
    id = db.Column(db.Integer, primary_key=True)
    slang_sentence = db.Column(db.String(256), nullable=True)
    standard_sentence = db.Column(db.String(256), nullable=True)


@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def data():
    # Query all users from the database
    data = Data.query.limit(100).all()
    return render_template('data.html', datas=data)

@app.route('/bleu-score/<int:id>',methods=["GET", "POST"])
def bleuScore(id):
    data = History.query.get(id)
    if request.method == "POST":
        reference_sentence = request.form.get("reference_sentence") 
       
        translated_sentence = translate_sentence(data.slang_sentence)
       
        bleu_score = compute_bleu_score(reference_sentence, translated_sentence)
       
        # TODO: Database Insert
        data.bleu_score = bleu_score
        data.reference_sentence = reference_sentence
        db.session.commit()
        
        return render_template('bleu-score.html', data=data,)

    return render_template('bleu-score.html', data=data)

@app.route('/history')
def history():
    # Query all users from the database
    history = History.query.all()
    return render_template('history.html', history=history)

@app.route('/program',methods=["GET", "POST"])
def program():
    if request.method == "POST":
       
        slang_sentence = request.form.get("slang_sentence") 
    
        translated_sentence = translate_sentence(slang_sentence)
 
 
        # TODO: Database Insert
        new_history = History(
            slang_sentence = slang_sentence,
            translate_sentence = translated_sentence,
        )
        db.session.add(new_history)
        db.session.commit()
        
        return render_template("translation.html", slang_sentence=slang_sentence, translated_sentence=translated_sentence)
    
    return render_template("translation.html")

with app.app_context():
    # Create the tables
    db.create_all()

if __name__ == "__main__":
    app.app_context()
    app.run(debug=True)
