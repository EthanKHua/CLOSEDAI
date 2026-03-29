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