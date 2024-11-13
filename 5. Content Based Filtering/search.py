import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

df = pd.read_csv('../Dataset/Item_data2.csv')


nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    if pd.isna(text) or isinstance(text, (int, float)):
        return ''
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]
    return ' '.join(tokens)


def get_synonyms(term):
    synonyms = set()
    for synset in nltk.corpus.wordnet.synsets(term):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def expand_query_with_synonyms(query):
    query_terms = query.lower().split()
    expanded_terms = query_terms.copy()
    for term in query_terms:
        expanded_terms.extend(get_synonyms(term))
    return " ".join(set(expanded_terms))


def search_products(train_data, query, top_n=10, 
                    exact_match_weight=1.0, 
                    ai_vector_weight=0.9, 
                    category_weight=0.7, 
                    word_search_weight=0.5, 
                    fuzzy_weight=0.3):
    
    query_processed = preprocess_text(query)
    query_expanded = expand_query_with_synonyms(query_processed)
    query_terms = query_expanded.split()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    train_data['Combined_Text'] = (
        train_data['Name'].fillna('') + ' ' +
        train_data['Description'].fillna('') + ' ' +
        train_data['Tags'].fillna('') + ' ' +
        train_data['Brand'].fillna('') + ' ' +
        train_data['Category'].fillna('')
    )

    # Step 1: Exact Match
    exact_matches = pd.DataFrame()
    for term in query_terms:
        exact_match = train_data[train_data['Name'].str.lower().str.contains(term, case=False, na=False)]
        exact_matches = pd.concat([exact_matches, exact_match]).drop_duplicates()
    
    exact_matches['Score'] = exact_match_weight if not exact_matches.empty else 0
    combined_matches = exact_matches if not exact_matches.empty else pd.DataFrame()

    # Step 2: AI Vector Search (TF-IDF with Cosine Similarity)
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Combined_Text'])
    input_vector = tfidf_vectorizer.transform([query_expanded])
    cosine_sim = cosine_similarity(input_vector, tfidf_matrix)[0]
    
    ai_vector_matches = train_data.copy()
    ai_vector_matches['Similarity'] = cosine_sim
    ai_vector_matches = ai_vector_matches[ai_vector_matches['Similarity'] > 0]
    ai_vector_matches['Score'] = ai_vector_matches['Similarity'] * ai_vector_weight
    
    combined_matches = pd.concat([combined_matches, ai_vector_matches]).drop_duplicates()

    # Step 3: Category Match
    category_matches = train_data[train_data['Category'].str.lower().str.contains(query_processed, case=False, na=False)]
    category_matches['Score'] = category_weight if not category_matches.empty else 0
    combined_matches = pd.concat([combined_matches, category_matches]).drop_duplicates()

    # Step 4: Word Search
    word_search_matches = pd.DataFrame()
    for term in query_terms:
        word_match = train_data[
            train_data['Name'].str.lower().str.contains(term, case=False, na=False) |
            train_data['Brand'].str.lower().str.contains(term, case=False, na=False) |
            train_data['Tags'].str.lower().str.contains(term, case=False, na=False) |
            train_data['Description'].str.lower().str.contains(term, case=False, na=False) |
            train_data['Category'].str.lower().str.contains(term, case=False, na=False)
        ]
        word_search_matches = pd.concat([word_search_matches, word_match]).drop_duplicates()
        
    word_search_matches['Score'] = word_search_weight if not word_search_matches.empty else 0
    combined_matches = pd.concat([combined_matches, word_search_matches]).drop_duplicates()

    # Step 5: Fuzzy Search
    fuzzy_matches = pd.DataFrame()
    for column in ['Name', 'Brand', 'Tags', 'Description', 'Category']:
        candidates = train_data[column].fillna('').astype(str)
        for candidate in candidates:
            score = fuzz.token_sort_ratio(query_processed, candidate)
            if score >= 80:
                match = train_data[train_data[column] == candidate]
                match['Score'] = score / 100 * fuzzy_weight
                fuzzy_matches = pd.concat([fuzzy_matches, match]).drop_duplicates()

    combined_matches = pd.concat([combined_matches, fuzzy_matches]).drop_duplicates()

    # Calculate final score and rank by content-based score
    combined_matches['Final_Score'] = combined_matches.groupby('ProdID')['Score'].transform('sum')
    results = combined_matches.sort_values(by='Final_Score', ascending=False).drop_duplicates('ProdID').head(top_n)

    return results[['ProdID', 'Name', 'Rating', 'Brand', 'Price', 'Tags']]


st.title("Product Search System")

# Input field for search query
query = st.text_input("Enter search query:")

if st.button("Search"):
    if query:
        results = search_products(df, query)
        st.write("Search Results:")
        st.dataframe(results)
    else:
        st.write("Please enter a search query.")


# to run : streamlit run search.py