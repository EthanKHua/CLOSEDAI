import numpy as np
import csv
from tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier


# constants
PAINTING_NAMES = ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']

# words filtered out before building bag of words
STOP_WORDS = {
    'the','a','an','and','or','but','in','on','of','to','is','it',
    'this','that','with','for','as','was','are','at','be','by','i',
    'me','my','its','so','have','has','had','not','do','did',
    'from','they','we','you','he','she','very','just','like','feel',
    'makes','make','feels','painting','art','piece','would'
}

# column rename map
COLUMN_MAP = {
    "unique_id":                                                                        "id",
    "Painting":                                                                         "painting",
    "On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?":    "emotion",
    "Describe how this painting makes you feel.":                                       "description",
    "This art piece makes me feel sombre.":                                             "sombre",
    "This art piece makes me feel content.":                                            "content",
    "This art piece makes me feel calm.":                                               "calm",
    "This art piece makes me feel uneasy.":                                             "uneasy",
    "How many prominent colours do you notice in this painting?":                       "num_colours",
    "How many objects caught your eye in the painting?":                                "num_objects",
    "How much (in Canadian dollars) would you be willing to pay for this painting?":    "price",
    "If you could purchase this painting, which room would you put that painting in?":  "room",
    "If you could view this art in person, who would you want to view it with?":        "view_with",
    "What season does this art piece remind you of?":                                   "season",
    "If this painting was a food, what would be?":                                      "as_food",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.": "soundtrack",
}

# feature groups
NUMERIC_COLS = ['emotion', 'sombre', 'content', 'calm', 'uneasy', 'num_colours', 'num_objects']
CAT_COLS     = ['room', 'view_with', 'season']
TEXT_COLS    = ['description', 'as_food', 'soundtrack']

# parsing helpers
def _word_counter(text):
    """
    Tokenise a raw string into a {word: count} dict.
    from colab pipeline:
        str.strip().lower().replace(',','').split(' ')  then Counter
    Uses a plain dict instead of Counter since Counter not allowed.
    """
    if not isinstance(text, str):
        return {}
    tokens = text.strip().lower().replace(',', '').split(' ')
    counts = {}
    for tok in tokens:
        if (
            tok and
            tok not in STOP_WORDS and
            len(tok) > 1
        ):
            counts[tok] = counts.get(tok, 0) + 1

    return counts
 
def _extract_numeric(rows, col):
    """
    Pull a float value from a column that may contain Likert text like
    '(3) Neutral'.  Extracts the first integer found.
    
    Returns np.nan on failure.
    """
    out = []
    for row in rows:
        val = row.get(col, '')
        m   = re.search(r'\d+', str(val))
        out.append(float(m.group()) if m else np.nan)
    return np.array(out, dtype=float)

def _read_csv_as_dicts(csv_filename):
    """
    Read the CSV using only stdlib csv.
    Rename columns via COLUMN_MAP.
    Return a list of row dicts with short column names.
    """
    rows = []
    with open(csv_filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            renamed = {COLUMN_MAP.get(k, k): v for k, v in row.items()}
            rows.append(renamed)
    return rows


def vectorise(csv_filename, vec_params):
    """
    Convert CSV into a feature matrix X of shape (M, d).
 
    Same pipeline as the colab:
      1. Min-max scale numeric columns using training min/max/median
      2. One-hot encode categorical cols using training category lists
      3. TF-IDF encode text cols using training vocab + IDF weights
 
    All scaling/vocab parameters come from vec_params, which was saved
    from the training data in the colab so that test rows are transformed
    identically.
 
    csv_filename : str  – path to the test CSV
    vec_params   : dict – saved from the colab via save_vec_params()
    Returns      : ndarray (M, d)
    """
    rows = _read_csv_as_dicts(csv_filename)
    M    = len(rows)
 
    feature_blocks = []
 
    # numeric columns
    # each column min-max scaled to [0, 1]
    # missing values replaced with training-set median before scaling
    for col in NUMERIC_COLS:
        vals   = _extract_numeric(rows, col)
        median = vec_params['numeric_medians'][col]
        mn     = vec_params['numeric_mins'][col]
        mx     = vec_params['numeric_maxs'][col]
 
        # fill NaN with training median
        vals[np.isnan(vals)] = median
 
        # min-max scaling 
        if mx > mn:
            vals = (vals - mn) / (mx - mn)
        else:
            vals = np.zeros(M, dtype=float)
 
        feature_blocks.append(vals.reshape(M, 1))
 
    # categorical columns (one-hot)
    # each response if comma-split into set of categories
    # a 1 is emitted for each category observed in the training set that is present in the response
    # categories not seen in training are ignored
    for col in CAT_COLS:
        # tokenise: split on comma, strip whitespace
        parsed = []
        for row in rows:
            raw  = row.get(col, '')
            cats = {c.strip() for c in str(raw).split(',') if c.strip() != 'nan'}
            parsed.append(cats)
 
        # emit one binary column per category observed in the training set
        cats_list = vec_params['cat_categories'][col]   # ordered list from training
        block     = np.zeros((M, len(cats_list)), dtype=float)
        for j, cat in enumerate(cats_list):
            for i, cat_set in enumerate(parsed):
                if cat in cat_set:
                    block[i, j] = 1.0
        feature_blocks.append(block)
 
    # text columns (TF-IDF)
    # for each word in the training vocab, TF-IDF score = tf * idf
    # IDF was computed on the training set and stored in vec_params
    # words not in the training vocab are ignored
    for col in TEXT_COLS:
        # tokenise 
        parsed = [_word_counter(row.get(col, '')) for row in rows]
 
        vocab_words = vec_params['tfidf_vocab'][col]    # ordered word list
        idf_weights = vec_params['tfidf_idf'][col]      # parallel IDF floats
 
        block = np.zeros((M, len(vocab_words)), dtype=float)
        for j, (word, idf) in enumerate(zip(vocab_words, idf_weights)):
            for i, d in enumerate(parsed):
                tf = float(d.get(word, 0))
                if tf > 0:
                    block[i, j] = tf * idf
        feature_blocks.append(block)
 
    return np.hstack(feature_blocks)


def build_model(data):
    estimators = []
    for arr in data:
        estimators.append(build_model_from_array(arr))
    
    rf = RandomForestClassifier()
    rf.estimators = estimators

    return rf

def build_model_from_array(arr):
    """
    From the provided array, build and return a fitted DecisionTreeClassifier. Refer to
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    and the Google Colab for more info
    """
    tree_nodes = [DecisionTreeClassifier() for _ in range(arr[0])]

    stack = [0]
    while stack:
        curr_node_index = stack.pop()
        curr_node = tree_nodes[curr_node_index]
        # if leaf node, update just the value
        if arr[1][curr_node_index] == -1 and arr[2][curr_node_index] == -1:
            curr_node.pred = np.argmax(arr[5][curr_node_index])
            continue

        curr_node.left = tree_nodes[arr[1][curr_node_index]]
        stack.append(arr[1][curr_node_index])

        curr_node.right = tree_nodes[arr[2][curr_node_index]]
        stack.append(arr[2][curr_node_index])

        curr_node.feature = arr[3][curr_node_index]
        curr_node.threshold = arr[4][curr_node_index]




if __name__ == "__main__":
    data = np.load("parameters.npy", allow_pickle=True)
    model = build_model_from_array(data)