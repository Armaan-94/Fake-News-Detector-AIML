from flask import Flask, render_template, request
import joblib
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model_welfake.pkl')
vectorizer = joblib.load('tfidf_vectorizer_welfake.pkl')

# Preprocess function (same as in your original code)
def preprocess_text(text):
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'])
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the article from the form input
    article = request.form['article']
    
    # Preprocess the article
    article_preprocessed = preprocess_text(article)
    
    # Transform the article using the TF-IDF vectorizer
    article_tfidf = vectorizer.transform([article_preprocessed])
    
    # Predict the label (Real or Fake)
    prediction = model.predict(article_tfidf)
    
    # Map the prediction to "Real" or "Fake"
    result = 'Real' if prediction[0] == 0 else 'Fake'
    
    return render_template('index.html', prediction=result, article=article)

if __name__ == '__main__':
    app.run(debug=True)