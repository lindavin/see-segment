import numpy as np

from see.base_classes import algorithm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer


class ClassifierFitness(algorithm):
    """Contains functions to return result of fitness function.
    and run classifier algorithm.

    Attributes
    ----------
    metric : string, default='accuracy'
        The metric to be used to test the classifier. 
        For a list of metrics, check out https://scikit-learn.org/stable/modules/model_evaluation.html.
        Examples include accuracy, balanced_accuracy, f1, roc_auc, etc...

    Methods
    -------
    evaluate(predictions, targets)
        Returns the error rate (i.e fitness) of the classifier according to the chosen metric.

    pipe(data)
        Evaluates the classifier on the dataset as the final stage
        of the classifier pipeline.
    """

    def __init__(self, paramlist=None, metric="accuracy"):
        """Generate algorithm params from parameter list."""
        super(ClassifierFitness, self).__init__(paramlist)
        self.metric = metric

    def evaluate(self, data):
        """
        Returns the error rate/fitness score of predictions.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        fitness : float
            The fitness score of the classifier (data.clf) after
            trained on the training set and tested on the testing
            set.

        Notes
        -----
        This method should be overridden by subclasses.
        """

        if data.testing_set is None:
            raise ValueError("Testing set cannot be none")
        if len(data.testing_set.X) <= 0:
            raise ValueError("Testing set must have at least one item")
        clf = data.clf

        clf.fit(data.training_set.X, data.training_set.y)

        scorer = get_scorer(self.metric)

        return 1 - scorer(clf, data.testing_set.X, data.testing_set.y)

    def pipe(self, data):
        """
        Evaluates the classifier on the dataset as the final stage
        of the classifier pipeline.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        data : PipelineClassifyDataset
            Attaches the fitness score to the data object.

        Notes
        -----
        Unless there is good reason to, one should not override this
        method.
        """

        if data.clf is None:
            raise Exception(
                "ERROR: classifier cannot be None. This must be set prior in the pipeline"
            )

        data.fitness = self.evaluate(data)

        return data

class F1Score(ClassifierFitness):
    def __init__(self, paramlist=None):
        super(F1Score, self).__init__(paramlist=paramlist, metric='f1')
        
class ROC_AUC(ClassifierFitness):
    def __init__(self, paramlist=None):
        super(ROC_AUC, self).__init__(paramlist=paramlist, metric='roc_auc')

class BalancedAccuracy(ClassifierFitness):
    def __init__(self, paramlist=None):
        super(BalancedAccuracy, self).__init__(paramlist=paramlist, metric='balanced_accuracy')

class CVFitness(ClassifierFitness):
    """Uses the Stratified Cross-Validaiton scheme to measure
    the fitness of a classifier algorithm.

    Attributes
    ----------
    cv : int, default=class.cv; read about more this in the Notes section.
        The number of folds to split the dataset.

    Methods
    -------
    set_cv(cv)
        Class method that sets the cv class attribute.

    pipe_evaluate(predictions, targets)
        Returns the average cross validation error
        of the classifier (data.clf).

    Notes
    -----
    The default cv class attribute is 10. To change this use
    the class method CVFitness#set_cv(cv).
    
    When this class is used during the classifier pipeline (i.e. as the 
    last item of a Workflow), the class attribute cv will be
    used to initialize this fitness instance.
    """

    cv = 10

    def __init__(self, paramlist=None, cv=None, metric='accuracy'):
        super(CVFitness, self).__init__(paramlist=paramlist, metric=metric)
        if cv is None:
            self.cv = CVFitness.cv
        else:
            self.cv = cv

    def evaluate(self, data):
        """
        Determines the fitness value of the attached classifier.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        data : PipelineClassifyDataset
        """

        if data.training_set is None:
            raise ValueError("Training set cannot be none")
        if len(data.training_set.X) <= 0:
            raise ValueError("Training set must have at least one item")

        scores = cross_val_score(
            data.clf, data.training_set.X, data.training_set.y, cv=self.cv, scoring=self.metric
        )
        data.scores = scores
        cv_fitness = 1 - scores.mean()
        return cv_fitness

    @classmethod
    def set_cv(clf, cv):
        """
        Class method that sets the cv class attribute.

        Parameters
        ----------
        cv : int
            The number of folds to split a dataset.

        Side Effects
        ------------
        Sets the class attribute cv. This should be done only once at the beginning.
        Instances of this class will use the class cv attribute to determine the
        number of splits to use for cross validation.

        Returns
        -------
        None
        """

        if type(cv) != int:
            raise ValueError("cv must be an int")

        clf.cv = cv
