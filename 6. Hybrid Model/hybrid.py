import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the SVD model
with open('hybrid_svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)

# Load data and compute cosine similarity
# Ensure 'df' is loaded with all required columns, including 'Combined_Text'
df = pd.read_csv('../Dataset/Item_data2.csv')  # Replace 'data.csv' with your actual data file

# Compute Combined_Text if not done
if 'Combined_Text' not in df.columns:
    df['Combined_Text'] = (
        df['Name'].fillna('') + ' ' +
        df['Description'].fillna('') + ' ' +
        df['Tags'].fillna('') + ' ' +
        df['Brand'].fillna('') + ' ' +
        df['Category'].fillna('')
    )

# Compute cosine similarity if not available
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tf.fit_transform(df['Combined_Text'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Helper functions
def map_product_indices(data):
    data = data.reset_index()
    indices = pd.Series(data.index, index=data['Name']).drop_duplicates()
    id_map = data[['ProdID', 'Name']].set_index('Name')
    indices_map = id_map.set_index('ProdID')
    return indices, indices_map

indices, indices_map = map_product_indices(df)

def hybrid_recommendation(UserID, product_name, indices, data_cleaned, cosine_sim, svd, display=10):
    if product_name in indices:
        idx = indices[product_name]
    else:
        raise ValueError(f"Product '{product_name}' not found in indices.")
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]

    max_index = len(data_cleaned) - 1
    product_indices = [i[0] for i in sim_scores if i[0] <= max_index]
    
    if not product_indices:
        raise ValueError(f"No valid product indices found for '{product_name}'.")
    
    products = df.iloc[product_indices][['ProdID', 'Name', 'Rating', 'Price']]
    products['est'] = products['ProdID'].apply(lambda x: svd.predict(UserID, x).est)
    
    products = products.sort_values('est', ascending=False)
    
    return products.head(display), product_indices

# Streamlit UI
st.title("Hybrid-Based Recommendation System")

# Populate dropdowns with all available user IDs and products
user_ids = df['ID'].unique()
product_names = df['Name'].unique()

user_id = st.selectbox("Select User ID:", options=user_ids)
product_name = st.selectbox("Select Product Name:", options=product_names)

if st.button("Get Recommendations"):
    try:
        recommendations, _ = hybrid_recommendation(
            user_id, product_name, indices, df, cosine_sim, svd, display=10
        )
        st.write("Top Recommendations:")
        st.dataframe(recommendations)
    except ValueError as e:
        st.write(str(e))


# To run : streamlit run hybrid.py
