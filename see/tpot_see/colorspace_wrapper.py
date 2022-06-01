from see.ColorSpace import colorspace, color_params
import numpy as np

class TPOTColorSpaceWrapper(colorspace):
    """Wrapper for the see.colorspace class so that it can work within tpot. Implements the
    fit and transform methods so that it can be used as a transformer in a scikit-learn
    pipeline, which is needed for tpot. Also uses keyword arguments for instantiation,
    so that segmentors can be properly instantiated by tpot.
    """
    
    def __init__(self, **kwargs):
        super().__init__(None)
        self.param_args = {}
        for key, value in kwargs.items():
            if(color_params.ranges[key] is not None):
                self.params[key] = value
                self.param_args[key] = value
            else:
                raise ValueError(f"{key} must have a corresponding value range in color_params")
            
    def fit(self, img_array, y=None):
        """Fits the algorithm to the data.

        Args:
            img_array (array-like): Array of images
            y (_type_, optional): _description_. Defaults to None.
        """
        pass

    def transform(self, img_array):
        """Transforms the array of images

        Args:
            img_array (array-like): Array of images

        Returns:
            Array of processed images
        """
        return np.array(list(map(self.evaluate, img_array)))
    
    def fit_transform(self, img_array, y=None):
        """Convenience function to fit and transform.

        Args:
            img_array (array-like): Array of images
            y (_type_, optional): Defaults to None.

        Returns:
            Array of processed images
        """
        self.fit(img_array, y)
        return self.transform(img_array)
