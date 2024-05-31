from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import Counter
from functools import lru_cache
import operator as optr
import concurrent.futures
import sqlite3

app = Flask(__name__)

# Function to load data from SQLite database
def load_data_from_db():
    conn = sqlite3.connect('words_database.db')
    
    jang_df = pd.read_sql_query("SELECT * FROM jang", conn)
    wordlist_df = pd.read_sql_query("SELECT * FROM wordlist", conn)
    
    conn.close()
    
    return jang_df['Sentence'].tolist(), wordlist_df['wordlist'].tolist()

# Load data from the SQLite database
jang_data, word_list_data = load_data_from_db()

# Preload data to minimize I/O
unigram_count = Counter([token for sentence in jang_data for token in sentence.split()])
bigram_count = Counter(zip([token for sentence in jang_data for token in sentence.split()],
                           [token for sentence in jang_data for token in sentence.split()][1:]))
words_list = set(word_list_data)

# Cache the minimum edit distance calculations to avoid redundant work
@lru_cache(maxsize=None)
def calculate_minimum_edit_distance(str1, str2):
    ic, dc, sc = 1, 1, 1
    n, m = len(str1), len(str2)
    med_dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        med_dp[i][0] = med_dp[i - 1][0] + dc
    for i in range(1, m + 1):
        med_dp[0][i] = med_dp[0][i - 1] + ic
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                med_dp[i][j] = min(med_dp[i - 1][j] + dc, med_dp[i][j - 1] + ic, med_dp[i - 1][j - 1])
            else:
                med_dp[i][j] = min(med_dp[i - 1][j] + dc, med_dp[i][j - 1] + ic, med_dp[i - 1][j - 1] + sc)
    return med_dp[n][m]

def get_candidate_words(err_word, med, candidates):
    candidate_words = set()
    for word in candidates:
        if calculate_minimum_edit_distance(err_word, word) <= med:
            candidate_words.add(word)
    return candidate_words

def calculate_bigram_probability(err_word, med, unigram_count, bigram_count, words_list, misspelled_previous, max_suggestions=3):
    uni_lambda = 0.4
    bi_lambda = 0.6
    candidate_word_probability_dict = {}
    candidate_words = get_candidate_words(err_word, med, words_list)
    prev_word = misspelled_previous.get(err_word, '')

    for word in candidate_words:
        unigram_probability = unigram_count.get(word, 0) / sum(unigram_count.values())
        bigram_probability = bigram_count.get((prev_word, word), 0) / unigram_count.get(prev_word, 1)
        candidate_word_probability = unigram_probability * uni_lambda + bigram_probability * bi_lambda
        candidate_word_probability_dict[word] = candidate_word_probability

    sorted_candidates = sorted(candidate_word_probability_dict.items(), key=optr.itemgetter(1), reverse=True)
    return dict(sorted_candidates[:max_suggestions])

def spell_correction(user_input):
    error_words = user_input.split()
    misspelled_previous = {}

    def process_word(word):
        med = 2
        return word, calculate_bigram_probability(word, med, unigram_count, bigram_count, words_list, misspelled_previous)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_word, error_words)

    found_idx = {word: result for word, result in results if result}
    return found_idx

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_us():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/correct', methods=['POST'])
def correct():
    user_input = request.form['user_input']
    suggestions = spell_correction(user_input)
    return render_template('result.html', user_input=user_input, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)

