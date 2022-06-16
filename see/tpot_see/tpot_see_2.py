import inspect
from see.ColorSpace import colorspace
from see.Segmentors import segmentor
from .base import TPOTBase
from .segment_fitness_wrapper import FitnessFunction

seg_config_dict = {
    # segmentation
    "see.tpot_see.segmentors_wrapper.TPOTSegmentorWrapper": {
        "algorithm": ["ColorThreshold"],
        "alpha1": [float(i) / 256 for i in range(0, 256)],
        "alpha2": [float(i) / 256 for i in range(0, 256)],
        "beta1": [float(i) / 256 for i in range(0, 256)],
        "beta2": [float(i) / 256 for i in range(0, 256)],
        "gamma1": [float(i) / 256 for i in range(0, 256)],
        "gamma2": [float(i) / 256 for i in range(0, 256)],
    },
    # TODO Search space is dummied up for testing
    "skimage.segmentation.felzenszwalb": {
        "scale": [1],
        "sigma": [100],
        "min_size": [1000],
        "multichannel": [True, False],
    },
    "skimage.segmentation.slic": {
        "n_segments": [3],
        "compactness": [3],
        "max_num_iter": [100],
        "sigma": [100],
        "convert2lab": [True, False],
        "multichannel": [True, False],
        "slic_zero": [True, False],
    },
    "skimage.segmentation.chan_vese": {
        "mu": [1],
        "lambda1": [1],
        "lambda2": [1],
        "max_num_iter": [100],
        "dt": [1],
        "init_level_set": ["checkerboard"],
    },
    "skimage.segmentation.morphological_chan_vese": {
        "smoothing": [1],
        "lambda1": [1],
        "lambda2": [1],
    },
    # preprocessing
    "see.tpot_see.colorspace_wrapper.TPOTColorSpaceWrapper": {
        "colorspace": [
            "RGB",
            "HSV",
            "RGB CIE",
            "XYZ",
            "YUV",
            "YIQ",
            "YPbPr",
            "YCbCr",
            "YDbDr",
        ],
        "multichannel": [True, False],
        "channel": [0, 1, 2],
    },
    "skimage.filters.gaussian": {
        "sigma": [1],
        "mode": ["reflect", "constant", "nearest", "mirror", "wrap"],
        "multichannel": [True, False],
        "preserve_range": [True, False],
    },
    # TODO: multichannel should probably be set in TPOTSegmentor
    "skimage.filters.unsharp_mask": {
        "radius": [1.0],
        "amount": [1.0],
        "multichannel": [True, False],
        "preserve_range": [True, False],
    }
    # What do we do with features?
    # "skimage.feature.blob_dog": {
    #     "min_sigma": [1],
    #     "max_sigma": [50],
    #     "sigma_ratio": [1.6],
    # },
    # "skimage.feature.graycoprops": {
    #     "prop": ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    # }
}


class TPOTSegmentor(TPOTBase):
    """TPOT estimator for image segmentation workflows."""

    scoring_function = FitnessFunction  # Segmentation scoring
    default_config_dict = seg_config_dict  # Segmentation dictionary
    classification = True
    regression = False

    def _check_dataset(self, features, target, sample_weight=None):
        if target is None:
            return features
        return features, target

    def _init_pretest(self, features, target):
        self.pretest_X = features
        self.pretest_y = target

    def _operator_class_checker(self, class_profile, op_obj):
        """Read operator_class_checker parameter of TPOTOperatorClassFactory from the .operator_utils.py file.

        Args:
            class_profile (Operator): Profile of the operator
            op_obj (object): Class of the operator

        Side effect:
            May update class_profile.

        Returns:
            string: The type of operator; must one of the main types in tpot.
        """
        from skimage import segmentation, filters, exposure, color, feature

        name = op_obj.__name__
        if hasattr(segmentation, name) or (
            inspect.isclass(op_obj) and issubclass(op_obj, segmentor)
        ):
            class_profile["root"] = True
            return "Classifier"
        elif (
            hasattr(filters, name)
            or hasattr(exposure, name)
            or hasattr(color, name)
            or hasattr(feature, name)
            or (inspect.isclass(op_obj) and issubclass(op_obj, colorspace))
        ):
            print('transformer')
            return "Transformer"
        else:
            print("WARNING Unknown")
