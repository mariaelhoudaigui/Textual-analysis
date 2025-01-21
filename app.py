from flask import Flask, render_template, request,send_file,jsonify
import nltk
import os
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from fpdf import FPDF
import spacy

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # configurer l'app flassk pour qu'elle utilise le dossier que vous avez définie comme repertoire de telechargement


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon') #téléchargez le lexique nécessaire pour utiliser le module SentimentIntensityAnalyzer de NLTK
sid = SentimentIntensityAnalyzer()

# Charger le modèle SpaCy pour le français
nlp = spacy.load('fr_core_news_sm')

# Fonctions utilitaires
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def get_tokens(text):
    return nltk.word_tokenize(text.lower(), language='french')

def summarize_text(text, max_sentences=3):
    stop_words = set(stopwords.words('french'))

    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text, language='french')

    # Tokenize the text into words and calculate word frequencies
    words = nltk.word_tokenize(text, language='french')
    words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    freq_table = Counter(words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = nltk.word_tokenize(sentence, language='french')
        for word in sentence_words:
            if word.lower() in freq_table:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq_table[word.lower()]
                else:
                    sentence_scores[sentence] += freq_table[word.lower()]

    # Get the highest-scoring sentences for the summary
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    summary = ' '.join(summary_sentences)
    return summary

def count_occurrences(text, phrase):
    phrase_tokens = get_tokens(phrase)
    text_tokens = get_tokens(text)
    text_length = len(text_tokens)
    phrase_length = len(phrase_tokens)
    count = 0

    # Rechercher la phrase dans le texte
    for i in range(text_length - phrase_length + 1):
        if text_tokens[i:i + phrase_length] == phrase_tokens:
            count += 1

    return count


#
def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    return scores

def plot_similarity(similarity_percentage_nltk, similarity_spacy):
    labels = ['Similarité(signification)', 'Similarité(mots communs)']
    values = [similarity_percentage_nltk, similarity_spacy]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['lightblue', 'darkblue'])
    plt.title('Comparaison de Similarité entre les Textes')
    plt.xlabel('Méthode de Similarité')
    plt.ylabel('Pourcentage de Similarité')
    plt.ylim(0, 100)  # Limite de l'axe y de 0 à 100%
    plt.grid(True)
    plt.savefig('static/similarity_plot.png')  # Sauvegarde du graphique dans un dossier statique
    plt.close()



# Function to create a PDF document
def create_pdf(comparison_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_text_color(0, 0, 255)  # Bleu
    pdf.multi_cell(200, 10, txt="Résultat de la Comparaison", align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln()
    pdf.ln()
    pdf.multi_cell(200, 10, txt=f"Texte 1 : {comparison_results['text1'][:500]}...", align='L')
    pdf.ln()
    pdf.multi_cell(200, 10, txt=f"Texte 2 : {comparison_results['text2'][:500]}...", align='L')
    pdf.ln()
    pdf.ln()


    pdf.set_text_color(0, 0, 255)  # Bleu
    pdf.multi_cell(200, 10, txt="Pourcentage_comparaison", align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(200, 10, txt=f"Similarité par signification : {comparison_results['similarity_spacy']}%", align='L')
    pdf.multi_cell(200, 10, txt=f"Pourcentage de Similarité : {comparison_results['similarity_percentage_nltk']}%",
                   align='L')
    pdf.ln()

    pdf.set_text_color(0, 0, 255)  # Bleu
    pdf.multi_cell(200, 10, txt="Sentiment du Texte 1", align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(200, 10, txt=f"Positif : {comparison_results['sentiment1']['pos']}", align='L')
    pdf.multi_cell(200, 10, txt=f"Négatif : {comparison_results['sentiment1']['neg']}", align='L')
    pdf.multi_cell(200, 10, txt=f"Neutre : {comparison_results['sentiment1']['neu']}", align='L')

    pdf.ln()
    pdf.set_text_color(0, 0, 255)  # Bleu
    pdf.multi_cell(200, 10, txt="Sentiment du Texte 2", align='L')
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(200, 10, txt=f"Positif : {comparison_results['sentiment2']['pos']}", align='L')
    pdf.multi_cell(200, 10, txt=f"Négatif : {comparison_results['sentiment2']['neg']}", align='L')
    pdf.multi_cell(200, 10, txt=f"Neutre : {comparison_results['sentiment2']['neu']}", align='L')

    # Charger et afficher une image
    image_path = r"C:\Users\E7490\PycharmProjects\pythonProject\static\similarity_plot.png"

 # Remplacez par le chemin de votre image
    pdf.image(image_path, x=10, y=150, w=180)

    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], "comparison_results.pdf")
    pdf.output(pdf_output_path)
    return pdf_output_path
# Route pour la page d'accueil
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', comparison_results=None, summary_results=None, search_results=None)

# Route pour la comparaison de texte
@app.route('/compare', methods=['POST'])
def compare_text():
    comparison_results = None

    text1 = request.form.get('text1', '')
    text2 = request.form.get('text2', '')
    file1 = request.files.get('file1')
    file2 = request.files.get('file2')

    if file1:
        text1 = extract_text_from_pdf(file1)
    if file2:
        text2 = extract_text_from_pdf(file2)

    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)

    common_words = set(tokens1).intersection(set(tokens2))
    num_common_words = len(common_words)
    total_words = len(set(tokens1).union(set(tokens2)))
    similarity_percentage_nltk = (num_common_words / total_words) * 100
    similarity_percentage_nltk = round(similarity_percentage_nltk, 2)
    unique_to_text1 = set(tokens1) - set(tokens2)
    unique_to_text2 = set(tokens2) - set(tokens1)

    if not common_words:
        common_words = ["Aucun mot commun trouvé"]
    if not unique_to_text1:
        unique_to_text1 = ["Aucun mot unique trouvé dans le Texte 1"]
    if not unique_to_text2:
        unique_to_text2 = ["Aucun mot unique trouvé dans le Texte 2"]

    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity_spacy = doc1.similarity(doc2) * 100
    similarity_spacy = round(similarity_spacy, 2)

    # Analyser le sentiment pour chaque texte
    sentiment1 = analyze_sentiment(text1)
    sentiment2 = analyze_sentiment(text2)

    plot_similarity(similarity_percentage_nltk, similarity_spacy)
    comparison_results = {
        'text1': text1,
        'text2': text2,
        'similarity_percentage_nltk': similarity_percentage_nltk,
        'similarity_spacy': similarity_spacy,
        'common_words': ', '.join(common_words),
        'unique_to_text1':', '.join(unique_to_text1),
        'unique_to_text2': ', '.join(unique_to_text2),
        'sentiment1': sentiment1,
        'sentiment2': sentiment2
    }

    pdf_path = create_pdf(comparison_results)

    return render_template('index.html', comparison_results=comparison_results, summary_results=None,
                           search_results=None, pdf_path=pdf_path)


# Route pour le téléchargement du PDF
@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    pdf_path = request.args.get('pdf_path') # récupère la valeur du paramètre nommé 'pdf_path' à partir de la requête GET.
    return send_file(pdf_path, as_attachment=True)


# Route pour le résumé de texte
@app.route('/summarize', methods=['POST'])
def summarize_text_route():
    text_to_summarize = request.form.get('text_to_summarize', '')
    file_to_summarize = request.files.get('file_to_summarize')

    if file_to_summarize:
        text_to_summarize = extract_text_from_pdf(file_to_summarize)

    summary = summarize_text(text_to_summarize)

    summary_results = {
        'original_text': text_to_summarize,
        'summary': summary
    }

    return jsonify(summary_results)
# Route pour la recherche de mot ou phrase
@app.route('/search-word', methods=['POST'])
def search_word():
    text_to_search = request.form.get('text_to_search', '')
    word_to_find = request.form.get('word_to_find', '')
    file_to_search = request.files.get('file_to_search')

    if file_to_search:
        text_to_search = extract_text_from_pdf(file_to_search)

    occurrences = count_occurrences(text_to_search, word_to_find)

    if occurrences > 0:
        # Trouver la première occurrence et la marquer
        paragraph = text_to_search
        start_index = paragraph.lower().find(word_to_find.lower())
        end_index = start_index + len(word_to_find)

        if start_index != -1:
            # Souligner la première occurrence trouvée
            highlighted_text = f"{paragraph[:start_index]}<span class='highlight'>{paragraph[start_index:end_index]}</span>{paragraph[end_index:]}"

            search_results = {
                'text_to_search': text_to_search,
                'word_to_find': word_to_find,
                'occurrences': occurrences,
                'word_found': True,
                'highlighted_text': highlighted_text  # Passer le texte souligné à l'interface utilisateur
            }
    else:
        search_results = {
            'text_to_search': text_to_search,
            'word_to_find': word_to_find,
            'occurrences': occurrences,
            'word_found': False
        }

    return jsonify(search_results)



if __name__ == '__main__':
    app.run(debug=True)
