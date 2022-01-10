"""This runs unit tests for the fitness stage of the classifier pipeline."""

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score

from see.classifier_helpers.fetch_data import generate_tutorial_data
from see.classifier_helpers.helpers import generate_train_test_set

from see import classifiers
from see.classifier_fitness import ClassifierFitness, F1Score, ROC_AUC, BalancedAccuracy, CVFitness
import pytest

def test_gaussian_naive_bayes_defaults():
    """Unit test for Gaussian Naive Bayes classifer algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = GaussianNB()

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - accuracy_score(pipeline_ds.testing_set.y, clf.predict(pipeline_ds.testing_set.X)))

    clf_container = classifiers.GaussianNBContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        ClassifierFitness().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == expected_fitness[i]

def test_nearest_neighbor_defaults():
    """Unit test for Nearest Neighbors classifer algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - accuracy_score(pipeline_ds.testing_set.y, clf.predict(pipeline_ds.testing_set.X)))

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        ClassifierFitness().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == expected_fitness[i]

def test_decision_tree_defaults():
    """Unit test for Decision Tree classifier algorithm.
    Check if classifier container with default parameters 
    performs the same as running the corresponding sklearn algorithm
    with their default parameters."""

    # Generate dataset
    datasets = generate_tutorial_data()

    # TODO: Problem even though we set the random state,
    # reproducible results are not gauranteeed. This
    # unit test may sometimes fail or sometimes pass.
    
    random_state = 21
    clf = DecisionTree(random_state=random_state)

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - accuracy_score(pipeline_ds.testing_set.y, clf.predict(pipeline_ds.testing_set.X)))

    # Supply random state so that test passes
    clf_container = classifiers.DecisionTreeContainer(random_state=random_state)
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds, random_state=random_state)
        ClassifierFitness().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        print(pipeline_ds.clf.get_params())
        assert actual_fitness == expected_fitness[i]
        
def test_nearest_neighbor_f1_score():
    """Unit test for Nearest Neighbors classifer algorithm on the
    f1 score metric."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - f1_score(pipeline_ds.testing_set.y, clf.predict(pipeline_ds.testing_set.X)))

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        F1Score().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == expected_fitness[i]
        
def test_nearest_neighbor_roc_auc_score():
    """Unit test for Nearest Neighbors classifer algorithm on the
    roc_auc metric."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()
    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - roc_auc_score(pipeline_ds.testing_set.y, 
                                                  clf.predict_proba(pipeline_ds.testing_set.X)[:,1]) # According to sklearn documentation, we want the probability of the class with the greater label. Hence, 1 (as opposed to 0) in the binary case.
                               )

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        ROC_AUC().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == expected_fitness[i]
        
def test_nearest_neighbor_balanced_accuracy_score():
    """Unit test for Nearest Neighbors classifer algorithm on the
    balanced accuracy metric."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf.fit(pipeline_ds.training_set.X, pipeline_ds.training_set.y)  # Train classifier
        expected_fitness.append(1 - balanced_accuracy_score(pipeline_ds.testing_set.y, clf.predict(pipeline_ds.testing_set.X)))

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())
    
    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        BalancedAccuracy().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == expected_fitness[i]

def test_nearest_neighbor_cv10_accuracy_score():
    """Unit test for Nearest Neighbors classifer algorithm on the
    balanced accuracy metric."""

    # Generate dataset
    datasets = generate_tutorial_data()

    clf = KNeighborsClassifier()

    # manual sklearn categorizations
    expected_fitness = []
    for ds_name in datasets:
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        expected_fitness.append((1 - cross_val_score(clf, pipeline_ds.training_set.X, pipeline_ds.training_set.y, cv=10)).mean())

    clf_container = classifiers.KNeighborsContainer()
    
    # Check that default params are equal
    assert (clf_container.create_clf().get_params() == clf.get_params())

    # Check that the evaluate function works correctly
    for i, ds_name in enumerate(datasets):
        X, y = datasets[ds_name]
        pipeline_ds = generate_train_test_set(X, y)
        clf_container.pipe(pipeline_ds)
        CVFitness().pipe(pipeline_ds)
        actual_fitness = pipeline_ds.fitness
        assert actual_fitness == pytest.approx(expected_fitness[i])
        
