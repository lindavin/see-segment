from see.Segment_Fitness import FitnessFunction as fitness_function
import numpy as np
from see.tpot_see.segmentors_wrapper import TPOTSegmentorWrapper

def FitnessFunction(_, segmentor: TPOTSegmentorWrapper, img, ground_truth):
    """Return fitness function result from inferred and ground_truth.

    Parameters:
    segmentor: segmentor
    img: array-like, array of input images
    ground_truth: array-like, array of ground truth segmentation mask for each training image.

    Outputs:
    Accuracy -- accuracy value as float
    """
    segmentor.fit(img, ground_truth)
    inferred = segmentor.predict(img)
    val = (1 - np.array(list(map(fitness_function, inferred, ground_truth)))[0])
    return val[0]