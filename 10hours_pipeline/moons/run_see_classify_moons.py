"""
The purpose of this script is to run collect data on the performance of
SEE-Classify applied to the dataset used in Dhahri et al. 2019.
This script will generate a CSV file, where each line contains the following information:

<trial-number>,<generation-number>,<best-hof-fitness>

The default parameters for the Genetic Algorithm are
Population Size (--pop-size) = 10
Number of Generations (--num-gen) = 10
Number of Trials (--num-trials) = 100
"""

# Path hack so that we can import see library.
import sys, os
sys.path.insert(0, os.path.abspath('../../'))

import argparse
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from see import GeneticSearch
from see.base_classes import pipedata
from see.classifiers import Classifier
from see.classifier_fitness import ClassifierFitness, CVFitness
from see.classifier_helpers import helpers
from see.classifier_helpers.fetch_data import fetch_wisconsin_data
from sklearn.datasets import make_moons, make_circles, make_classification, fetch_openml
from see.Workflow import workflow

parser = argparse.ArgumentParser(description="Create csv data files. Cross-validation is not used by default.")

parser.add_argument(
    "--num-gen",
    default=100,
    type=int,
    help="number of generations to run genetic search (default: 20)",
)

parser.add_argument(
    "--pop-size",
    default=100,
    type=int,
    help="population size of each generation to run genetic search (default: 20)",
)

parser.add_argument(
    "--num-trials",
    default=100,
    type=int,
    help="number of trials to run genetic search (default: 100)",
)

parser.add_argument(
    "--fitness-func", "--metric",
    default='simple',
    choices=['simple', 'accuracy', 'f1', 'balanced_accuracy', 'roc_auc'],
    type=str,
    help="the metric for fitness function for the GA (default: simple). This can be either simple or cv10.",
)

parser.add_argument(
    "--cross-val",
    dest="cross_val",
    action="store_true",
    help="set pipeline to USE cross validation",
)

parser.add_argument(
    "--no-cross-val",
    dest="cross_val",
    action="store_false",
    help="set pipeline to NOT USE cross validation",
)

parser.set_defaults(cross_val=True)

args = parser.parse_args()

random.seed()

random_state = int(random.random()*1000000)

X, y  = make_moons(noise=0.3, random_state=random_state)

# Preprocess data
# not sure about this step....
# X = StandardScaler().fit_transform(X)


#random_state = 42
print("# random.getstate: {}".format(random.getstate()))

print("# random_state int: {}".format(random_state))

print("# Size of dataset: {}".format(len(X)))

# Split data into training, testing, and validation sets 
# A train-test-validation split of 75-25

temp = helpers.generate_train_test_set(X, y, test_size=0.25, random_state=random_state, stratify=y)
validation_set = temp.testing_set
print("# Size of validation set: {}".format(len(validation_set.X)))

if args.fitness_func == 'simple':
    # Update fitness metric to new system
    args.fitness_func = 'accuracy'

if args.cross_val:
    pipeline_dataset = temp # training data is used in pipeline
    CVFitness.set_cv(10)
    CVFitness.set_metric(args.fitness_func)
    fitness_func = CVFitness
    print("# USING Cross Validation")
    print("# Size of training data: {}".format(len(pipeline_dataset.training_set.X)))
    print("# Fitness Function: {}".format(args.fitness_func))
else:
    # split training data into training and testing sets
    pipeline_dataset = helpers.generate_train_test_set(temp.training_set.X, temp.training_set.y, test_size=0.25, random_state=random_state, stratify=temp.training_set.y)
    fitness_func = ClassifierFitness
    print("# NOT USING Cross Validation")
    print("# Size of training set: {}".format(len(pipeline_dataset.training_set.X)))
    print("# Size of testing set: {}".format(len(pipeline_dataset.testing_set.X)))
    print("# Fitness Function: {}".format(args.fitness_func))

print("\n")

# Initialize Algorithm Space and Workflow
# Classifier.use_dhahri_space()
# Use the entire default space

# Check algorithm space
algorithm_space = Classifier.algorithmspace
print("# Algorithm Space: ")
print(list(algorithm_space.keys()))
print("\n")


workflow.addalgos([Classifier, fitness_func])
wf = workflow()
NUM_GENERATIONS = args.num_gen
NUM_TRIALS = args.num_trials
POP_SIZE = args.pop_size

print("GA running for {} generations with population size of {}".format(NUM_GENERATIONS, POP_SIZE))

for i in range(NUM_TRIALS):
    print("Running trial number {}".format(i))
    my_evolver = GeneticSearch.Evolver(workflow, pipeline_dataset, pop_size=POP_SIZE)
    my_evolver.run(
        ngen=NUM_GENERATIONS,
	print_raw_data=True
    )

    my_hof = my_evolver.hof
    best_ind = my_hof[0]
    print('# BEST Individual: {}'.format(best_ind))
    fitness = best_ind.fitness.values[0]
    print('# Training fitness: {}'.format(fitness))

    # retrain classifier before testing validation

    algo_name = best_ind[0]
    param_list = best_ind
    clf = Classifier.algorithmspace[algo_name](param_list).create_clf()
    temp.clf = clf

    score = ClassifierFitness().evaluate(temp)

    print('# Validation score: {}'.format(score))
