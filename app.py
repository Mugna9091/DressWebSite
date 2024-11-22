from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from rembg import remove
from PIL import Image

app = Flask(__name__)
DATABASE_PATH = 'database/abbigliamento.json'

# Inizializza il modello AI
model = SentenceTransformer('all-MiniLM-L6-v2')


# Funzioni di utilità
def load_data():
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, 'r') as file:
            return json.load(file)
    return []


def save_data(image_path, description):
    # Calcola l'embedding della descrizione
    embedding = model.encode(description).tolist()

    # Salva nel database
    data = load_data()
    data.append({"image_path": image_path, "description": description, "embedding": embedding})
    with open(DATABASE_PATH, 'w') as file:
        json.dump(data, file)


def search_clothing(query):
    # Calcola l'embedding della query
    query_embedding = model.encode(query).reshape(1, -1)

    # Carica i dati e calcola la similarità
    data = load_data()
    results = []
    for item in data:
        db_embedding = item['embedding']
        similarity = cosine_similarity(query_embedding, [db_embedding])[0][0]
        results.append((item, similarity))

    # Ordina per somiglianza
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Ritorna i 20 risultati più simili
    return [r[0] for r in results[:20]]


def remove_background(image_file):

    # Apri l'immagine caricata
    input_image = image_file.read()

    # Rimuovi lo sfondo
    output = remove(input_image)

    # Salva l'immagine elaborata
    output_path = f"static/dress/{image_file.filename}"
    with open(output_path, 'wb') as f:
        f.write(output)

    return output_path


# Rotte
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/abbigliamento', methods=['GET', 'POST'])
def abbigliamento():
    if request.method == 'POST':
        query = request.form.get('search')
        results = search_clothing(query)
        return render_template('abbigliamento.html', results=results)
    else:
        data = load_data()
        return render_template('abbigliamento.html', data=data)


@app.route('/contatti')
def contatti():
    return render_template('contatti.html')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        file = request.files['image']
        description = request.form.get('description')

        if file and description:
            # Rimuove lo sfondo dall'immagine
            processed_image_path = remove_background(file)

            # Salva l'immagine processata e i dati nel database
            save_data(processed_image_path, description)
            return redirect(url_for('admin'))

    return render_template('admin.html')



if __name__ == '__main__':
    app.run(debug=True)
