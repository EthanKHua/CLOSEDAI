class RandomForestClassifier:
    n_estimators = None
    max_depth = None
    min_samples_split = None
    random_state = None
    criterion = None

    estimators = []

    def __init__(self, n_estimators = 100, max_depth = 20, min_samples_split = 2, random_state = 16, criterion = 'entropy'):
        """
        Initializes the random forest classifier. If explicit hyperparameters are not given, default to the tuned hyperparameters
        found the the colab
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.criterion = criterion
        self.estimators = []

    def predict(self, x):
        """
        Majority-vote prediction for a single sample x (shape (d,)).
        Each tree in self.estimators votes and the most common label gets selected.
        If tied, then select the lowest class index.
        """
        votes = {}
        for tree in self.estimators:
            label = tree.predict(x)
            votes[label] = votes.get(label, 0) + 1
        # return the label with the most votes
        return max(votes, key=lambda k: votes[k])

    def predict_all(self, X):
        """
        Run predict() on every row of X (shape (M, d)).
        Returns a list of predicted labels of length M.
        """
        return [self.predict(x) for x in X]