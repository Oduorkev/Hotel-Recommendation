from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load your dataset (replace 'your_data.csv' with your dataset file)
final_data = pd.read_csv('final.csv')

# Combine relevant features into a single column for TF-IDF vectorization
final_data['combined_features'] = final_data['features'] + ' ' + final_data['cuisine'] + ' ' + final_data['town'] + ' ' + final_data['name'] + ' ' + final_data['category']

# Fill missing values in the 'combined_features' column with an empty string
final_data['combined_features'] = final_data['combined_features'].fillna('')

# Create a TF-IDF vectorizer for the 'combined_features' column
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(final_data['combined_features'])

# Create a NearestNeighbors model
n_neighbors = 10  # Number of similar hotels to recommend
knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
knn_model.fit(tfidf_matrix)

# Function to get hotel recommendations
def recommend_hotels_restaurants(input_features, n_recommendations=10):
    # Transform input_features into a TF-IDF vector
    input_vector = tfidf_vectorizer.transform([input_features])

    # Find the most similar hotels
    _, indices = knn_model.kneighbors(input_vector, n_neighbors=n_recommendations)

    # Return the recommended hotels with relevant information
    recommended_hotels = final_data.iloc[indices[0]][['name','category', 'website', 'rating', 'phone', 'combined_features', 'locationString', 'average_price']]

    return recommended_hotels

# Function to get town-based recommendations
def recommend_town_hotels(town, n_recommendations=30):
    town = town.lower()  # Convert to lowercase for consistency

    # Filter hotels in the specified town
    townbase = final_data[final_data['town'].str.lower() == town]

    if not townbase.empty:
        # Sort the hotels by rating (high to low) and average_price (low to high)
        townbase = townbase.sort_values(by=['rating', 'average_price'], ascending=[False, True])

        # Select specific columns to display in the recommendation, including the new columns (Name, Category, Rating, Features, Website, Phone)
        recommended_hotels = townbase[['name', 'category', 'town', 'rating', 'combined_features', 'locationString', 'average_price', 'website', 'phone']]

        return recommended_hotels.head(n_recommendations)
    else:
        return None


towns = [
    'nairobi', 'kitengela', 'karen', 'syokimau', 'athi river', 'bomet',
    'eldoret', 'watamu', 'malindi', 'kikambala',
    'malindi marine national park', 'kakamega', 'nairobi region',
    'kiambu', 'lavington', 'thika', 'narok', 'ruaka', 'nyeri',
    'langata', 'kilifi', 'ngong', 'kiserian', 'ongata rongai',
    'kahawa', 'mombasa', 'shanzu', 'bamburi', 'mtwapa', 'likoni',
    'kitale', 'kwale', 'tiwi', 'lake elementaita', 'nakuru',
    'lake nakuru national park', 'lamu island', 'shela', 'matuu',
    'masii', 'mtito andei', 'nanyuki town', 'nyahururu', 'kikuyu',
    'limuru', 'siaya', 'juja', 'diani beach', 'ukunda', 'naivasha',
    'maasai mara national reserve', 'sekenani', 'kisumu', 'mambrui',
    'nanyuki municipality', 'embu', 'meru town', 'kisii', 'machakos',
    'naboisho conservancy', 'ololaimutiek', 'kajiado', 'migori',
    'tsavo national park west', 'ruiru', 'bungoma', 'isiolo',
    'kericho', 'tsavo', 'gilgil', 'galu beach', 'voi',
    'tsavo national park east', 'busia', 'kitui',
    'mara north conservancy', 'olderkesi private reserve', 'shella',
    'narasha', 'kuwinda', 'kwoyo', 'maai mahiu', 'talek', 'homa bay',
    'mount kenya national park', 'mbita', 'samburu national reserve',
    'mwingi', 'mlolongo', 'lodwar', 'sagana', 'bondo',
    'amboseli national park', 'naro moru'
]

@app.route('/')
def index():
    return render_template('index.html', towns=towns)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_input = request.form['user_input']
    recommended_hotels = recommend_hotels_restaurants(user_input, n_recommendations=5)
    return render_template('recommendations.html', recommendations=recommended_hotels)

@app.route('/town_recommendations', methods=['POST'])
def town_recommendations():
    selected_town = request.form['selected_town']
    town_recommendations = recommend_town_hotels(selected_town, n_recommendations=10)
    return render_template('town_recommendations.html', town=selected_town, recommendations=town_recommendations)



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)




