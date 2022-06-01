from see.Segmentors import segmentor, seg_params
import numpy as np
import sys

class TPOTSegmentorWrapper(segmentor):
    """Wrapper for the see.segmentor class so that it can work within tpot. Implements the
    fit and predict methods so that it can be used as an estimator in a scikit-learn
    pipeline, which is needed for tpot. Also uses keyword arguments for instantiation,
    so that segmentors can be properly instantiated by tpot.
    """
    
    def __init__(self, **kwargs):
        super().__init__(None)
        self.param_args = {}
        for key, value in kwargs.items():
            if(seg_params.ranges[key] is not None):
                self.params[key] = value
                self.param_args[key] = value
            else:
                raise ValueError(f"{key} must have a corresponding value range in seg_params")

    def evaluate(self, img):
        """Run segmentation algorithm to get inferred mask."""
        sys.stdout.flush()
        self.thisalgo = segmentor.algorithmspace[self.params['algorithm']](None)
        for key, value in self.param_args.items():
            # Set parameter values
            self.thisalgo[key] = value
        return self.thisalgo.evaluate(img)
    
    def fit(self, img_array, ground_truth):
        """Trains the segmentor.

        Args:
            img_array (array-like): Array of images
            ground_truth (array-like): Array of corresponding ground truth masks
        """
        pass
    
    def predict(self, img_array):
        """Method to allow segmentors to work in tpot.

        Args:
            img_array (array-like): Array of images

        Returns:
            Array of inferred masks for each image in img_array
        """
        return np.array(list(map(self.evaluate, img_array)))