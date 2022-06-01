from .base import TPOTBase
from .segment_fitness_wrapper import FitnessFunction

seg_config_dict = {
    'se.segmentor': {
        'algorithm': ['ColorThreshold', 'Felzenszwalb', 'Slic'],
        'alpha1': [float(i) / 256 for i in range(0, 256)],
        'alpha2': [float(i) / 256 for i in range(0, 256)],
        'beta1': [float(i) / 256 for i in range(0, 256)],
        'beta2': [float(i) / 256 for i in range(0, 256)],
        'gamma1': [float(i) / 256 for i in range(0, 256)],
        'gamma2': [float(i) / 256 for i in range(0, 256)],
        'n_segments': [i for i in range(0, 10)],
        'max_iter': [i for i in range(1, 20)]
    },
    'see.ColorSpace.colorspace': {
        'colorspace': [
        'RGB',
        'HSV',
        'RGB CIE',
        'XYZ',
        'YUV',
        'YIQ',
        'YPbPr',
        'YCbCr',
        'YDbDr'],
        'multichannel': [True, False],
        'channel': [0, 1, 2],
    }
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

    def operator_class_checker(self, class_profile, op_obj):
        from colorspace_wrapper import TPOTColorSpaceWrapper
        from segmentors_wrapper import TPOTSegmentorWrapper
        if issubclass(op_obj, TPOTColorSpaceWrapper):
            return "Transformer"
        elif issubclass(op_obj, TPOTSegmentorWrapper):
            class_profile["root"] = True
            return "Classifier"
        else:
            print("WARNING Unknown")