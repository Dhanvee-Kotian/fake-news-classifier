import os
import re
import joblib
import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. SETUP NLTK & PREPROCESSING ---
# Replicating the exact logic from your notebook
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Exact preprocessing function from the training notebook.
    Crucial for model accuracy.
    """
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words and len(word) > 2
    ]

    return ' '.join(words)

# --- 2. LOAD MODELS & VECTORIZER ---
print("⏳ Loading models and vectorizer...")

try:
    # Load Vectorizer
    tfidf = joblib.load('tfidf_vectorizer.pkl')

    # Load Best Model Name
    with open('best_model.txt', 'r') as f:
        best_model_name = f.read().strip()

    # Load All 5 Models
    # Names match the keys used in your notebook
    models = {
        'logistic_regression': joblib.load('logistic_regression_model.pkl'),
        'random_forest': joblib.load('random_forest_model.pkl'),
        'svm': joblib.load('svm_model.pkl'),
        'decision_tree': joblib.load('decision_tree_model.pkl'),
        'naive_bayes': joblib.load('naive_bayes_model.pkl')
    }
    print(f"✅ Loaded {len(models)} models. Champion: {best_model_name}")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Ensure you have run the Jupyter Notebook and generated the .pkl files.")
    models = {}
    best_model_name = "None"
    tfidf = None

# --- 3. ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not tfidf or not models:
            return render_template('index.html', error="Models not loaded correctly.")

        raw_text = request.form['news']
        cleaned_text = clean_text(raw_text)
        
        # Vectorize input
        # Note: transform() expects an iterable, so we wrap in list
        vec_text = tfidf.transform([cleaned_text])
        
        results = []
        
        # Run prediction on ALL models
        for name, model in models.items():
            # Get probability/confidence
            # SVM was trained with probability=True, so this works for all
            try:
                probs = model.predict_proba(vec_text)[0]
                confidence = max(probs)
                pred_idx = np.argmax(probs)
            except:
                # Fallback if predict_proba is not available
                pred_idx = model.predict(vec_text)[0]
                confidence = 1.0 # Default if probability unavailable
            
            prediction_label = "TRUE NEWS" if pred_idx == 1 else "FAKE NEWS"
            
            # Format name for display (e.g., 'random_forest' -> 'Random Forest')
            display_name = name.replace('_', ' ').title()
            
            results.append({
                "raw_name": name,
                "display_name": display_name,
                "prediction": prediction_label,
                "confidence": round(confidence * 100, 2),
                "is_best": (name == best_model_name)
            })

        # Separate best result from the rest
        best_result = next((r for r in results if r['is_best']), results[0])
        other_results = [r for r in results if not r['is_best']]

        # Extract Keywords (Top 5 features from TF-IDF for this input)
        feature_names = tfidf.get_feature_names_out()
        sorted_indices = vec_text.toarray()[0].argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in sorted_indices if vec_text.toarray()[0][i] > 0]

        return render_template('index.html', 
                             user_input=raw_text,
                             best_result=best_result,
                             other_results=other_results,
                             keywords=keywords)

    return render_template('index.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    message = None
    if request.method == 'POST':
        if 'dataset' not in request.files:
            message = "No file part"
        else:
            file = request.files['dataset']
            if file.filename == '':
                message = "No selected file"
            elif file and file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                message = f"Dataset '{file.filename}' uploaded successfully! ."
            else:
                message = "Invalid file type. Please upload a CSV."
                
    return render_template('admin.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)