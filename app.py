from flask import Flask, render_template, request
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__, template_folder="templates")

# Load model yang sudah disimpan
with open("model/model_numpy.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load data
data = pd.read_csv("data/data_clean.csv")

# Train SVD model
X_train = data[['ram_id', 'usage', 'range', 'display(in inch)', 'price(in Rs.)']]
svd = TruncatedSVD(n_components=3)
X_train_svd = svd.fit_transform(X_train)

def convert_rs_to_rupiah(amount_rs):
    kurs = 194.75
    amount_rupiah = amount_rs * kurs
    return format_rupiah(amount_rupiah)

def format_rupiah(amount):
    return f"Rp {amount:,.0f}".replace(",", ".")

def recommend_laptops(filter_ram, usage_filter, svd, X_train_svd):
    # Filter data berdasarkan kriteria
    filtered_data = data[(data['ram_id'] == filter_ram) & (data['usage'] == usage_filter)]

    if not filtered_data.empty:
        # Transformasi data menggunakan model SVD
        X_filtered = svd.transform(filtered_data[['ram_id', 'usage', 'range', 'display(in inch)', 'price(in Rs.)']])

        # Hitung cosine similarity
        cos_sim = cosine_similarity(X_train_svd, X_filtered)
  
        # Ambil indeks laptop dengan similarity tertinggi
        top_laptop_indices = cos_sim.argmax(axis=1)

        # Ambil laptop dengan similarity tertinggi
        recommended_laptops = []
        recommended_indices = set()  
        for index in top_laptop_indices:
            if index not in recommended_indices:
                recommended_indices.add(index) 
                laptop = filtered_data.iloc[index].copy()
                laptop['formatted_price'] = convert_rs_to_rupiah(laptop['price(in Rs.)'])
                recommended_laptops.append(laptop)

        if recommended_laptops:
            return recommended_laptops
        else:
            return []
    return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        print(request.form) 
        ram = int(request.form['ram_id'])
        usage = int(request.form['usage'])
        recommended_laptops = recommend_laptops(ram, usage, svd, X_train_svd)
        return render_template('recommend.html', laptops=recommended_laptops)

if __name__ == '__main__':
    app.run(debug=True)