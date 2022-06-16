from .operator_utils import ARGTypeClassFactory, Operator, ARGType, _is_resampler, source_decode, _is_transformer, _is_selector
import inspect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import types
import warnings

def isFunction(func):
    return type(func) == types.FunctionType

def FuncToConstructor(func):
    # constructor from function signature
    init = getattr(func, "deprecated_original", func)
    init_signature = inspect.signature(init)
    # TODO ignore callable parameters
    parameters = [
        p
        for p in init_signature.parameters.values()
        if (p.name != "image" and p.kind != p.VAR_KEYWORD and isFunction(p.default) != True )
    ]
    optionalQuoteString = lambda a: f"'{a}'" if isinstance(a, str) else a
    # TODO p.default repeated too many times
    param_str = ', '.join([f"{p.name}={None if p.default == p.empty else optionalQuoteString(p.default)}" for p in parameters])
    body_str = '\n'.join([f"\tself.{p.name}={p.name}" for p in parameters])
    loc = {}
    constructor_str = f"def constructor(self, {param_str}):\n{body_str}"
    exec(constructor_str, loc)
    return loc["constructor"]

# TODO Add the remaining types. for now this only implements Transformer, Classifier, Regressor so that it can quickly work.
def FunctionToSklearnClass(func, op_type):
    class_type = None
    class_profile = {}
    
    constructor = FuncToConstructor(func)
    class_profile["__init__"] = constructor
    
    # method
    def fit(self, x, y):
        pass
    
    def call_func(self, x):
        return np.array(list(map(lambda x1: func(x1, **self.get_params()), x)))
        
    # get all keyword arguments for a function
    if(op_type == "Classifier" or op_type == "Regressor"):
        class_profile["fit"] = fit
        class_profile["predict"] = call_func
        mixin = ClassifierMixin if op_type == "Classifier" else RegressorMixin
        class_type = type(f"{func.__name__.capitalize()}Segmentor", (BaseEstimator, mixin), class_profile)
    elif(op_type == "Transformer"):
        class_profile["fit"] = fit
        class_profile["transform"] = call_func
        class_type = type(f"{func.__name__.capitalize()}", (BaseEstimator, TransformerMixin), class_profile)
    else:
        raise ValueError(f"Unimplemented operator type: {op_type}")

    return class_type

def FunctionOperatorClassFactory(
    opsourse, opdict, BaseClass=Operator, ArgBaseClass=ARGType, verbose=0, operator_class_checker=None
):
    """Dynamically create operator class for a scikit-image image processing function.

    Parameters
    ----------
    opsourse: string
        operator source in config dictionary (key)
    opdict: dictionary
        operator params in config dictionary (value)
    BaseClass: Class
        inherited BaseClass for operator
    ArgBaseClass: Class
        inherited BaseClass for parameter
    verbose: int, optional (default: 0)
        How much information TPOT communicates while it's running.
        0 = none, 1 = minimal, 2 = high, 3 = all.
        if verbose > 2 then ImportError will raise during initialization
    operator_class_checker: Callable[[class_profile: Operator, operator_class: object], string]
        custom function to check if an operator can be the root of a pipeline.
        operator_class is the class of the operator and class_profile is the corresponding
        base class of the operator used in TPOT. May modify the Operator object and maps the 
        class to one of "Classifier", "Regressor", "Selector", or "Transformer"; should raise ValueError
        otherwise. This was added so that TPOT can work with SEE.

    Returns
    -------
    op_class: Class
        a new class for a operator
    arg_types: list
        a list of parameter class

    """
    class_profile = {}
    dep_op_list = {}  # list of nested estimator/callable function
    dep_op_type = {}  # type of nested estimator/callable function
    import_str, op_str, op_obj = source_decode(opsourse, verbose=verbose)

    if not op_obj:
        return None, None
    else:
        # define if the operator can be the root of a pipeline
        if operator_class_checker is not None:
            if(not (callable(operator_class_checker))):
                raise ValueError(
                    "operator_class_checker must be callable"
                )
            optype = operator_class_checker(class_profile, op_obj) # May need to raise error if not one of the main types
        else:
            if is_classifier(op_obj):
                class_profile["root"] = True
                optype = "Classifier"
            elif is_regressor(op_obj):
                class_profile["root"] = True
                optype = "Regressor"
            elif _is_selector(op_obj):
                optype = "Selector"
            elif _is_transformer(op_obj):
                optype = "Transformer"
            elif _is_resampler(op_obj):
                optype = "Resampler"
            else:
                raise ValueError(
                    "optype must be one of: Classifier, Regressor, Selector, Transformer, or Resampler"
                )
        
        @classmethod
        def op_type(cls):
            """Return the operator type.

            Possible values:
                "Classifier", "Regressor", "Selector", "Transformer"
            """
            return optype

        import_hash = {}
        import_hash[import_str] = [op_str]
        
        # we expect op_obj to be a callable function
        if(isFunction(op_obj)):
            # Build class from function
            op_obj = FunctionToSklearnClass(op_obj, optype)
            import_hash[import_str].append(f"{op_str}#auto_gen_for_funcs")
        else:
            warnings.warn("Warning, operator should be a function!")
        class_profile["type"] = op_type
        class_profile["sklearn_class"] = op_obj
        
        arg_types = []
        for pname in sorted(opdict.keys()):
            prange = opdict[pname]
            if not isinstance(prange, dict):
                classname = "{}__{}".format(op_str, pname)
                arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass))
            else:
                for dkey, dval in prange.items():
                    dep_import_str, dep_op_str, dep_op_obj = source_decode(
                        dkey, verbose=verbose
                    )
                    if dep_import_str in import_hash:
                        import_hash[dep_import_str].append(dep_op_str)
                    else:
                        import_hash[dep_import_str] = [dep_op_str]
                    dep_op_list[pname] = dep_op_str
                    dep_op_type[pname] = dep_op_obj
                    if dval:
                        for dpname in sorted(dval.keys()):
                            dprange = dval[dpname]
                            classname = "{}__{}__{}".format(op_str, dep_op_str, dpname)
                            arg_types.append(
                                ARGTypeClassFactory(classname, dprange, ArgBaseClass)
                            )
        class_profile["arg_types"] = tuple(arg_types)
        class_profile["import_hash"] = import_hash
        class_profile["dep_op_list"] = dep_op_list
        class_profile["dep_op_type"] = dep_op_type

        @classmethod
        def parameter_types(cls):
            """Return the argument and return types of an operator.

            Parameters
            ----------
            None

            Returns
            -------
            parameter_types: tuple
                Tuple of the DEAP parameter types and the DEAP return type for the
                operator

            """
            return ([np.ndarray] + arg_types, np.ndarray)  # (input types, return types)

        class_profile["parameter_types"] = parameter_types

        @classmethod
        def export(cls, *args):
            """Represent the operator as a string so that it can be exported to a file.

            Parameters
            ----------
            args
                Arbitrary arguments to be passed to the operator

            Returns
            -------
            export_string: str
                String representation of the sklearn class with its parameters in
                the format:
                SklearnClassName(param1="val1", param2=val2)

            """
            op_arguments = []

            if dep_op_list:
                dep_op_arguments = {}
                for dep_op_str in dep_op_list.values():
                    dep_op_arguments[dep_op_str] = []

            for arg_class, arg_value in zip(arg_types, args):
                aname_split = arg_class.__name__.split("__")
                if isinstance(arg_value, str):
                    arg_value = '"{}"'.format(arg_value)
                if len(aname_split) == 2:  # simple parameter
                    op_arguments.append("{}={}".format(aname_split[-1], arg_value))
                # Parameter of internal operator as a parameter in the
                # operator, usually in Selector
                else:
                    dep_op_arguments[aname_split[1]].append(
                        "{}={}".format(aname_split[-1], arg_value)
                    )

            tmp_op_args = []
            if dep_op_list:
                # To make sure the initial operators is the first parameter just
                # for better presentation
                for dep_op_pname, dep_op_str in dep_op_list.items():
                    arg_value = dep_op_str  # a callable function, e.g scoring function
                    doptype = dep_op_type[dep_op_pname]
                    if inspect.isclass(doptype):  # a estimator
                        if (
                            issubclass(doptype, BaseEstimator)
                            or is_classifier(doptype)
                            or is_regressor(doptype)
                            or _is_transformer(doptype)
                            or _is_resampler(doptype)
                            or issubclass(doptype, Kernel)
                        ):
                            arg_value = "{}({})".format(
                                dep_op_str, ", ".join(dep_op_arguments[dep_op_str])
                            )
                    tmp_op_args.append("{}={}".format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

        class_profile["export"] = export

        op_classname = "TPOT_{}".format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        return op_class, arg_types
