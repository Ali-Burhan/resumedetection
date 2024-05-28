from flask import render_template,Flask,request
import pickle
from PyPDF2 import PdfReader 
import re



app = Flask(__name__)

# load models
rf_classifier = pickle.load(open('models/rf_classifier.pkl','rb'))
tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl','rb'))


#Helper Function For Project ======================================>
def pdf_to_txt(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# method to clean resume text
def clean_resume(txt):
    cleanText = re.sub('http\S+\s'," ",txt)
    cleanText = re.sub('RT|cc'," ",cleanText)
    cleanText = re.sub('#\S+\s'," ",cleanText)
    cleanText = re.sub('@\S+'," ",cleanText)
    cleanText = re.sub('@\S+'," ",cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]'," ",cleanText)
    cleanText = re.sub('\s'," ",cleanText)
    return cleanText


#Main method to predict the result
def predict_cat(inp):
    resume_text = clean_resume(inp)
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    predict_category = rf_classifier.predict(resume_tfidf)[0]
    return predict_category


#All Files Routes =================================================>
@app.route('/')
def resume():
    return render_template('resume.html')

#Form method to get resume file
@app.route('/pred',methods=['POST','GET'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename
        if filename.endswith('.pdf'):
            text = pdf_to_txt(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            pass
        else:
            return render_template('resume.html',message="Invalid file format, Please provide pdf or txt file")
        predict_category = predict_cat(text)
        return render_template('resume.html',predicted_category = predict_category)


#main method to run flask
if __name__ == "__main__":
    app.run(debug=True)