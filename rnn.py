import numpy as np
from sklearn.neural_network import MLPClassifier

hidden_sizes = range(1, 21)
alphas = range(0, 2, 0.2)
learning_rates = [0.001, 0.01, 0.025, 0.05, 0.1]

for h in hidden_sizes:
    for alpha in alphas:
        for a in learning_rates:
            model = MLPClassifier(
                hidden_layer_sizes=(h,),
                solver='sgd',
                learning_rate_init=a,
                max_iter=200
            )
            model.fit(X_train, t_train)
            print(model.best_validation_score)