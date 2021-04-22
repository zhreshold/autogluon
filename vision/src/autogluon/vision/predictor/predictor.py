"""Image Prediction task"""
import copy
import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from gluoncv.auto.tasks import ImageClassification as _ImageClassification
from gluoncv.model_zoo import get_model_list

from autogluon.core.constants import MULTICLASS
from autogluon.core.data.label_cleaner import LabelCleaner
from autogluon.core.utils import set_logger_verbosity
from autogluon.core.utils import verbosity2loglevel, get_gpu_count
from ..configs.presets_configs import unpack, _check_gpu_memory_presets
from ..utils import MXNetErrorCatcher

__all__ = ['ImagePredictor']

logger = logging.getLogger()  # return root logger


class ImagePredictor(object):
    """AutoGluon Predictor for predicting image category based on their whole contents

    Parameters
    ----------
    problem_type : str, default = None
        Type of prediction problem. Options: ('multiclass'). If problem_type = None, the prediction problem type is inferred
         based on the provided dataset. Currently only multiclass(or single class vs. background) classification is supported.
    eval_metric : str, default = None
        Metric by which to evaluate the data with. Options: ('accuracy').
        Currently only supports accuracy for multiclass classification.
    path : str, default = None
        The directory for saving logs or intermediate data. If unspecified, will create a sub-directory under
        current working directory.
    verbosity : int, default = 2
        Verbosity levels range from 0 to 4 and control how much information is printed.
        Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
        If using logging, you can alternatively control amount of information printed via logger.setLevel(L),
        where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)
    """
    # Dataset is a subclass of `pd.DataFrame`, with `image` and `label` columns.
    Dataset = _ImageClassification.Dataset

    def __init__(self, problem_type=None, eval_metric=None, path=None, verbosity=2):
        self._problem_type = problem_type
        self._eval_metric = eval_metric
        if path is None:
            path = os.getcwd()
        self._log_dir = path
        self._verbosity = verbosity
        self._classifier = None
        self._label_cleaner = None
        self._fit_summary = {}
        os.makedirs(self._log_dir, exist_ok=True)

    @property
    def path(self):
        return self._log_dir

    @unpack('image_predictor')
    def fit(self,
            train_data,
            tuning_data=None,
            time_limit='auto',
            presets=None,
            hyperparameters=None,
            **kwargs):
        """Automatic fit process for image prediction.

        Parameters
        ----------
        train_data : pd.DataFrame or str
            Training data, can be a dataframe like image dataset.
            For dataframe like datasets, `image` and `label` columns are required.
            `image`: raw image paths. `label`: categorical integer id, starting from 0.
            For more details of how to construct a dataset for image predictor, check out:
            `http://preview.d2l.ai/d8/main/image_classification/getting_started.html`.
            If a string is provided, will search for d8 built-in datasets.
        tuning_data : pd.DataFrame or str, default = None
            Another dataset containing validation data reserved for model selection and hyperparameter-tuning,
            can be a dataframe like image dataset.
            If a string is provided, will search for k8 datasets.
            If `None`, the validation dataset will be randomly split from `train_data` according to `holdout_frac`.
        time_limit : int, default = 'auto'(defaults to 2 hours if no presets detected)
            Time limit in seconds, if `None`, will run until all tuning and training finished.
            If `time_limit` is hit during `fit`, the
            HPO process will interrupt and return the current best configuration.
        presets : list or str or dict, default = ['medium_quality_faster_train']
            List of preset configurations for various arguments in `fit()`. Can significantly impact predictive accuracy, memory-footprint, and inference latency of trained models,
            and various other properties of the returned `predictor`.
            It is recommended to specify presets and avoid specifying most other `fit()` arguments or model hyperparameters prior to becoming familiar with AutoGluon.
            As an example, to get the most accurate overall predictor (regardless of its efficiency), set `presets='best_quality'`.
            To get good quality with faster inference speed, set `presets='good_quality_faster_inference'`
            Any user-specified arguments in `fit()` will override the values used by presets.
            If specifying a list of presets, later presets will override earlier presets if they alter the same argument.
            For precise definitions of the provided presets, see file: `autogluon/vision/configs/presets_configs.py`.
            Users can specify custom presets by passing in a dictionary of argument values as an element to the list.
            Available Presets: ['best_quality', 'high_quality_fast_inference', 'good_quality_faster_inference', 'medium_quality_faster_train']
            It is recommended to only use one `quality` based preset in a given call to `fit()` as they alter many of the same arguments and are not compatible with each-other.

            Note that depending on your specific hardware limitation(# gpu, size of gpu memory...) your mileage may vary a lot, you may choose lower quality presets if necessary, and
            try to reduce `batch_size` if OOM("RuntimeError: CUDA error: out of memory") happens frequently during the `fit`.

            In-depth Preset Info:
                best_quality={
                    'hyperparameters': {
                        'model': Categorical('resnet50_v1b', 'resnet101_v1d', 'resnest200'),
                        'lr': Real(1e-5, 1e-2, log=True),
                        'batch_size': Categorical(8, 16, 32, 64, 128),
                        'epochs': 200,
                        'early_stop_patience': -1
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 1024,
                        'search_strategy': 'bayesopt'},
                    'time_limit': 12*3600,}
                    Best predictive accuracy with little consideration to inference time or model size. Achieve even better results by specifying a large time_limit value.
                    Recommended for applications that benefit from the best possible model accuracy.

                good_quality_fast_inference={
                    'hyperparameters': {
                        'model': Categorical('resnet50_v1b', 'resnet34_v1b'),
                        'lr': Real(1e-4, 1e-2, log=True),
                        'batch_size': Categorical(8, 16, 32, 64, 128),
                        'epochs': 150,
                        'early_stop_patience': 20
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 512,
                        'search_strategy': 'bayesopt'},
                    'time_limit': 8*3600,}
                    Good predictive accuracy with fast inference.
                    Recommended for applications that require reasonable inference speed and/or model size.

                medium_quality_faster_train={
                    'hyperparameters': {
                        'model': 'resnet50_v1b',
                        'lr': 0.01,
                        'batch_size': 64,
                        'epochs': 50,
                        'early_stop_patience': 5
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 8,
                        'search_strategy': 'random'},
                    'time_limit': 1*3600,}

                    Medium predictive accuracy with very fast inference and very fast training time.
                    This is the default preset in AutoGluon, but should generally only be used for quick prototyping.

                medium_quality_faster_inference={
                    'hyperparameters': {
                        'model': Categorical('resnet18_v1b', 'mobilenetv3_small'),
                        'lr': Categorical(0.01, 0.005, 0.001),
                        'batch_size': Categorical(64, 128),
                        'epochs': Categorical(50, 100),
                        'early_stop_patience': 10
                        },
                    'hyperparameter_tune_kwargs': {
                        'num_trials': 32,
                        'search_strategy': 'bayesopt'},
                        'time_limit': 2*3600,}

                    Medium predictive accuracy with very fast inference.
                    Comparing with `medium_quality_faster_train` it uses faster model but explores more hyperparameters.
        hyperparameters : dict, default = None
            Extra hyperparameters for specific models.
            Accepted args includes(not limited to):
            epochs : int, default value based on network
                The `epochs` for model training.
            net : mx.gluon.Block
                The custom network. If defined, the model name in config will be ignored so your
                custom network will be used for training rather than pulling it from model zoo.
            optimizer : mx.Optimizer
                The custom optimizer object. If defined, the optimizer will be ignored in config but this
                object will be used in training instead.
            batch_size : int
                Mini batch size
            lr : float
                Trainer learning rate for optimization process.
            early_stop_patience : int, default=10
                Number of epochs with no improvement after which train is early stopped. Use `None` to disable.
            early_stop_min_delta : float, default=1e-4
                The small delta value to ignore when evaluating the metric. A large delta helps stablize the early
                stopping strategy against tiny fluctuation, e.g. 0.5->0.49->0.48->0.499->0.500001 is still considered as
                a good timing for early stopping.
            early_stop_baseline : float, default=None
                The minimum(baseline) value to trigger early stopping. For example, with `early_stop_baseline=0.5`,
                early stopping won't be triggered if the metric is less than 0.5 even if plateau is detected.
                Use `None` to disable.
            early_stop_max_value : float, default=None
                The max value for metric, early stop training instantly once the max value is achieved. Use `None` to disable.
            You can get the list of accepted hyperparameters in `config.yaml` saved by this predictor.
        **kwargs :
            holdout_frac : float, default = 0.1
                The random split ratio for `tuning_data` if `tuning_data==None`.
            random_state : int, default = None
                The random_state(seed) for shuffling data, only used if `tuning_data==None`.
                Note that the `random_state` only affect the splitting process, not model training.
                If not specified(None), will leave the original random sampling intact.
            nthreads_per_trial : int, default = (# cpu cores)
                Number of CPU threads for each trial, if `None`, will detect the # cores on current instance.
            ngpus_per_trial : int, default = (# gpus)
                Number of GPUs to use for each trial, if `None`, will detect the # gpus on current instance.
            hyperparameter_tune_kwargs: dict, default = None
                num_trials : int, default = 1
                    The limit of HPO trials that can be performed within `time_limit`. The HPO process will be terminated
                    when `num_trials` trials have finished or wall clock `time_limit` is reached, whichever comes first.
                search_strategy : str, default = 'random'
                    Searcher strategy for HPO, 'random' by default.
                    Options include: ‘random’ (random search), ‘bayesopt’ (Gaussian process Bayesian optimization),
                    ‘grid’ (grid search).
                max_reward : float, default = None
                    The reward threashold for stopping criteria. If `max_reward` is reached during HPO, the scheduler
                    will terminate earlier to reduce time cost.
                scheduler_options : dict, default = None
                    Extra options for HPO scheduler, please refer to :class:`autogluon.core.Searcher` for details.
        """
        if self._problem_type is None:
            # options: multiclass
            self._problem_type = MULTICLASS
        assert self._problem_type in (MULTICLASS,), f"Invalid problem_type: {self._problem_type}"
        if self._eval_metric is None:
            # options: accuracy,
            self._eval_metric = 'accuracy'

        # init/validate kwargs
        kwargs = self._validate_kwargs(kwargs)
        # unpack
        num_trials = kwargs['hyperparameter_tune_kwargs']['num_trials']
        nthreads_per_trial = kwargs['nthreads_per_trial']
        ngpus_per_trial = kwargs['ngpus_per_trial']
        holdout_frac = kwargs['holdout_frac']
        random_state = kwargs['random_state']
        scheduler = kwargs['hyperparameter_tune_kwargs']['scheduler']
        searcher = kwargs['hyperparameter_tune_kwargs']['searcher']
        max_reward = kwargs['hyperparameter_tune_kwargs']['max_reward']
        scheduler_options = kwargs['hyperparameter_tune_kwargs']['scheduler_options']
        # deep copy to avoid inplace overwrite
        train_data = copy.deepcopy(train_data)
        tuning_data = copy.deepcopy(tuning_data)

        log_level = verbosity2loglevel(self._verbosity)
        set_logger_verbosity(self._verbosity, logger=logger)
        if presets:
            if not isinstance(presets, list):
                presets = [presets]
            logger.log(20, f'Presets specified: {presets}')

        if time_limit == 'auto':
            # no presets, no user specified time_limit
            time_limit = 7200
            logger.log(20, f'`time_limit=auto` set to `time_limit={time_limit}`.')

        use_rec = False
        if isinstance(train_data, str) and train_data == 'imagenet':
            logger.warning('ImageNet is a huge dataset which cannot be downloaded directly, ' +
                           'please follow the data preparation tutorial in GluonCV.' +
                           'The following record files(symlinks) will be used: \n' +
                           'rec_train : ~/.mxnet/datasets/imagenet/rec/train.rec\n' +
                           'rec_train_idx : ~/.mxnet/datasets/imagenet/rec/train.idx\n' +
                           'rec_val : ~/.mxnet/datasets/imagenet/rec/val.rec\n' +
                           'rec_val_idx : ~/.mxnet/datasets/imagenet/rec/val.idx\n')
            train_data = pd.DataFrame({'image': [], 'label': []})
            tuning_data = pd.DataFrame({'image': [], 'label': []})
            use_rec = True
        if isinstance(train_data, str):
            from d8.image_classification import Dataset as D8D
            names = D8D.list()
            if train_data.lower() in names:
                train_data = D8D.get(train_data)
            else:
                valid_names = '\n'.join(names)
                raise ValueError(f'`train_data` {train_data} is not among valid list {valid_names}')
            if tuning_data is None:
                train_data, tuning_data = train_data.split(1 - holdout_frac)
        if isinstance(tuning_data, str):
            from d8.image_classification import Dataset as D8D
            names = D8D.list()
            if tuning_data.lower() in names:
                tuning_data = D8D.get(tuning_data)
            else:
                valid_names = '\n'.join(names)
                raise ValueError(f'`tuning_data` {tuning_data} is not among valid list {valid_names}')

        # data sanity check
        train_data = self._validate_data(train_data)
        train_labels = _get_valid_labels(train_data)
        self._label_cleaner = LabelCleaner.construct(problem_type=self._problem_type, y=train_labels, y_uncleaned=train_labels)
        train_labels_cleaned = self._label_cleaner.transform(train_labels)
        # converting to internal label set
        _set_valid_labels(train_data, train_labels_cleaned)
        if tuning_data is not None:
            tuning_data = self._validate_data(tuning_data)
            _set_valid_labels(tuning_data, self._label_cleaner.transform(_get_valid_labels(tuning_data)))

        if self._classifier is not None:
            logging.getLogger("ImageClassificationEstimator").propagate = True
            self._classifier._logger.setLevel(log_level)
            self._fit_summary = self._classifier.fit(train_data, tuning_data, 1 - holdout_frac, random_state, resume=False)
            if hasattr(self._classifier, 'fit_history'):
                self._fit_summary['fit_history'] = self._classifier.fit_history()
            return self

        # new HPO task
        if time_limit is not None and num_trials is None:
            num_trials = 99999
        if time_limit is None and num_trials is None:
            raise ValueError('`time_limit` and `num_trials` can not be `None` at the same time, '
                             'otherwise the training will not be terminated gracefully.')
        config = {'log_dir': self._log_dir,
                  'num_trials': 99999 if num_trials is None else max(1, num_trials),
                  'time_limits': 2147483647 if time_limit is None else max(1, time_limit),
                  'searcher': searcher,
                  # needed for gluon-cv TODO: remove after gluon-cv is updated https://github.com/dmlc/gluon-cv/issues/1633
                  'search_strategy': searcher,
                  'scheduler': scheduler,
                  }
        if max_reward is not None:
            config['max_reward'] = max_reward
        if nthreads_per_trial is not None:
            config['nthreads_per_trial'] = nthreads_per_trial
        if ngpus_per_trial is not None:
            config['ngpus_per_trial'] = ngpus_per_trial
        if isinstance(hyperparameters, dict):
            if 'batch_size' in hyperparameters:
                bs = hyperparameters['batch_size']
                _check_gpu_memory_presets(bs, ngpus_per_trial, 4, 256)  # 256MB per sample
            net = hyperparameters.pop('net', None)
            if net is not None:
                config['custom_net'] = net
            optimizer = hyperparameters.pop('optimizer', None)
            if optimizer is not None:
                config['custom_optimizer'] = optimizer
            # check if hyperparameters overwriting existing config
            for k, v in hyperparameters.items():
                if k in config:
                    raise ValueError(f'Overwriting {k} = {config[k]} to {v} by hyperparameters is ambiguous.')
            config.update(hyperparameters)
        if scheduler_options is not None:
            config.update(scheduler_options)
        if use_rec == True:
            config['use_rec'] = True
        if 'early_stop_patience' not in config:
            config['early_stop_patience'] = 10
        if config['early_stop_patience'] == None:
            config['early_stop_patience'] = -1
        # TODO(zhreshold): expose the transform function(or sign function) for converting custom metrics
        if 'early_stop_baseline' not in config or config['early_stop_baseline'] == None:
            config['early_stop_baseline'] = -np.Inf
        if 'early_stop_max_value' not in config or config['early_stop_max_value'] == None:
            config['early_stop_max_value'] = np.Inf
        # verbosity
        if log_level > logging.INFO:
            logging.getLogger('gluoncv.auto.tasks.image_classification').propagate = False
            logging.getLogger("ImageClassificationEstimator").propagate = False
            logging.getLogger("ImageClassificationEstimator").setLevel(log_level)
        task = _ImageClassification(config=config)
        # GluonCV can't handle these separately - patching created config
        task.search_strategy = scheduler
        task.scheduler_options['searcher'] = searcher
        task._logger.setLevel(log_level)
        task._logger.propagate = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MXNetErrorCatcher() as err:
                self._classifier = task.fit(train_data, tuning_data, 1 - holdout_frac, random_state)
            if err.exc_value is not None:
                raise RuntimeError(err.exc_value + err.hint)
        self._classifier._logger.setLevel(log_level)
        self._classifier._logger.propagate = True
        self._fit_summary = task.fit_summary()
        if hasattr(task, 'fit_history'):
            self._fit_summary['fit_history'] = task.fit_history()
        return self

    def _validate_data(self, data):
        """Check whether data is valid, try to convert with best effort if not"""
        if not (hasattr(data, 'classes') and hasattr(data, 'to_mxnet')):
            if isinstance(data, pd.DataFrame):
                # raw dataframe, try to add metadata automatically
                if 'label' in data.columns and 'image' in data.columns:
                    # check image relative/abs path is valid
                    sample = data.iloc[0]['image']
                    if not os.path.isfile(sample):
                        raise OSError(f'Detected invalid image path `{sample}`, please ensure all image paths are absolute or you are using the right working directory.')
                    logger.log(20, 'Converting raw DataFrame to ImagePredictor.Dataset...')
                    infer_classes = sorted(data.label.unique().tolist())
                    logger.log(20, f'Detected {len(infer_classes)} unique classes: {infer_classes}')
                    instruction = 'train_data = ImagePredictor.Dataset(train_data, classes=["foo", "bar"])'
                    logger.log(20, f'If you feel the `classes` is inaccurate, please construct the dataset explicitly, e.g. {instruction}')
                    data = _ImageClassification.Dataset(data, classes=infer_classes)
                else:
                    err_msg = 'Unable to convert raw DataFrame to ImagePredictor Dataset, ' + \
                              '`image` and `label` columns are required.' + \
                              'You may visit `https://auto.gluon.ai/stable/tutorials/image_prediction/dataset.html` ' + \
                              'for details.'
                    raise AttributeError(err_msg)
            else:
                raise TypeError(f"Unable to process dataset of type: {type(data)}")
        elif isinstance(data, _ImageClassification.Dataset):
            assert 'label' in data.columns
            assert hasattr(data, 'classes')
            # check whether classes are outdated, no action required if all unique labels is subset of `classes`
            unique_labels = sorted(data['label'].unique().tolist())
            if not (all(ulabel in data.classes for ulabel in unique_labels)):
                data.classes = unique_labels
                logger.log(20, f'Reset labels to {unique_labels}')
        if len(data) < 1:
            raise ValueError('Empty dataset.')
        return data

    def _validate_kwargs(self, kwargs):
        """validate and initialize default kwargs"""
        kwargs['holdout_frac'] = kwargs.get('holdout_frac', 0.1)
        if not (0 < kwargs['holdout_frac'] < 1.0):
            raise ValueError(f'Range error for `holdout_frac`, expected to be within range (0, 1), given {kwargs["holdout_frac"]}')
        kwargs['random_state'] = kwargs.get('random_state', None)
        kwargs['nthreads_per_trial'] = kwargs.get('nthreads_per_trial', None)
        kwargs['ngpus_per_trial'] = kwargs.get('ngpus_per_trial', None)
        if kwargs['ngpus_per_trial'] is not None and kwargs['ngpus_per_trial'] > 0:
            detected_gpu = get_gpu_count()
            if detected_gpu < kwargs['ngpus_per_trial']:
                raise ValueError(f"Insufficient detected # gpus {detected_gpu} vs requested {kwargs['ngpus_per_trial']}")
        # tune kwargs
        hpo_tune_args = kwargs.get('hyperparameter_tune_kwargs', {})
        hpo_tune_args['num_trials'] = hpo_tune_args.get('num_trials', 1)
        hpo_tune_args['searcher'] = hpo_tune_args.get('searcher', 'random')
        if not hpo_tune_args['searcher'] in ('random', 'bayesopt', 'grid'):
            raise ValueError(f"Invalid searcher: {hpo_tune_args['searcher']}, supported: ('random', 'bayesopt', 'grid')")
        hpo_tune_args['scheduler'] = hpo_tune_args.get('scheduler', 'local')
        if not hpo_tune_args['scheduler'] in ('fifo', 'local'):
            raise ValueError(f"Invalid searcher: {hpo_tune_args['searcher']}, supported: ('fifo', 'local')")
        hpo_tune_args['max_reward'] = hpo_tune_args.get('max_reward', None)
        if hpo_tune_args['max_reward'] is not None and hpo_tune_args['max_reward'] < 0:
            raise ValueError(f"Expected `max_reward` to be a positive float number between 0 and 1.0, given {hpo_tune_args['max_reward']}")
        hpo_tune_args['scheduler_options'] = hpo_tune_args.get('scheduler_options', None)
        kwargs['hyperparameter_tune_kwargs'] = hpo_tune_args
        return kwargs

    def predict_proba(self, data, as_pandas=True):
        """Predict images as a whole, return the probabilities of each category rather
        than class-labels.

        Parameters
        ----------
        data : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or list of numpy array(s) (False).
            Pandas object is a DataFrame.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain probs of each category. If more than one image in input,
            the returned dataframe will contain `images` column, and all results are concatenated.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier is not initialized, try `fit` first.')
        assert self._label_cleaner is not None
        y_pred_proba = self._classifier.predict(data, with_proba=True)
        if isinstance(data, pd.DataFrame) and 'image' in data:
            idx_to_image_map = data[['image']]
            idx_to_image_map = idx_to_image_map.reset_index(drop=False)
            y_pred_proba = idx_to_image_map.merge(y_pred_proba, on='image')
            y_pred_proba = y_pred_proba.set_index('index').rename_axis(None)
        y_pred_proba[list(self._label_cleaner.cat_mappings_dependent_var.values())] = y_pred_proba['image_proba'].to_list()
        ret = y_pred_proba.drop(['image', 'image_proba'], axis=1, errors='ignore')

        if as_pandas:
            return ret
        else:
            return ret.to_numpy()

    def predict(self, data, as_pandas=True):
        """Predict images as a whole, return labels(class category).

        Parameters
        ----------
        data : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or list of numpy array(s) (False).
            Pandas object is a DataFrame.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain labels. If more than one image in input,
            the returned dataframe will contain `images` column, and all results are concatenated.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier is not initialized, try `fit` first.')
        assert self._label_cleaner is not None
        proba = self._classifier.predict(data)
        if 'image' in proba.columns:
            # multiple images
            assert isinstance(data, pd.DataFrame) and 'image' in data.columns
            y_pred = proba.loc[proba.groupby(["image"])["score"].idxmax()].reset_index(drop=True)
            idx_to_image_map = data[['image']]
            idx_to_image_map = idx_to_image_map.reset_index(drop=False)
            y_pred = idx_to_image_map.merge(y_pred, on='image')
            y_pred = y_pred.set_index('index').rename_axis(None)
            ret = self._label_cleaner.inverse_transform(y_pred['id'].rename('label'))
        else:
            # single image
            ret = proba.loc[[proba["score"].idxmax()]]
            ret = self._label_cleaner.inverse_transform(ret['id'].rename('label'))
        if as_pandas:
            return ret
        else:
            return ret.to_numpy()

    def predict_feature(self, data, as_pandas=True):
        """Predict images visual feature representations, return the features as numpy (1xD) vector.

        Parameters
        ----------
        data : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.
        as_pandas : bool, default = True
            Whether to return the output as a pandas object (True) or list of numpy array(s) (False).
            Pandas object is a DataFrame.

        Returns
        -------

        pd.DataFrame
            The returned dataframe will contain image features. If more than one image in input,
            the returned dataframe will contain `images` column, and all results are concatenated.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier is not initialized, try `fit` first.')
        ret = self._classifier.predict_feature(data)
        if as_pandas:
            return ret
        else:
            return ret.to_numpy()

    def evaluate(self, data):
        """Evaluate model performance on validation data.

        Parameters
        ----------
        data : pd.DataFrame or iterator
            The validation data.
        """
        if self._classifier is None:
            raise RuntimeError('Classifier not initialized, try `fit` first.')
        return self._classifier.evaluate(data)

    def fit_summary(self):
        """Return summary of last `fit` process.

        Returns
        -------
        dict
            The summary of last `fit` process. Major keys are ('train_acc', 'val_acc', 'total_time',...)

        """
        return copy.copy(self._fit_summary)

    def save(self, path=None):
        """Dump predictor to disk.

        Parameters
        ----------
        path : str, default = None
            The path of saved copy. If not specified(None), will automatically save to `self.path` directory
            with filename `image_predictor.ag`

        """
        if path is None:
            path = os.path.join(self.path, 'image_predictor.ag')
        with open(path, 'wb') as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, path, verbosity=2):
        """Load previously saved predictor.

        Parameters
        ----------
        path : str
            The file name for saved pickle file. If `path` is a directory, will try to load the file `image_predictor.ag` in
            this directory.
        verbosity : int, default = 2
            Verbosity levels range from 0 to 4 and control how much information is printed.
            Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings).
            If using logging, you can alternatively control amount of information printed via logger.setLevel(L),
            where L ranges from 0 to 50 (Note: higher values of L correspond to fewer print statements, opposite of verbosity levels)

        """
        if os.path.isdir(path):
            path = os.path.join(path, 'image_predictor.ag')
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
        obj._verbosity = verbosity
        return obj

    @classmethod
    def list_models(cls):
        """Get the list of supported model names in model zoo that
        can be used for image classification.

        Returns
        -------
        tuple of str
            A tuple of supported model names in str.

        """
        return tuple(_SUPPORTED_MODELS)


def _get_valid_labels(data):
    ret = None
    if isinstance(data, pd.DataFrame):
        ret = data['label']
    else:
        from d8.image_classification import Dataset as D8D
        if isinstance(data, D8D):
            ret = data.df['class_name']
    if ret is None:
        raise ValueError('Dataset must be pandas.DataFrame or d8.image_classification.Dataset')
    return ret

def _set_valid_labels(data, label):
    if isinstance(data, pd.DataFrame):
        data['label'] = label
    else:
        from d8.image_classification import Dataset as D8D
        if isinstance(data, D8D):
            data.df['class_name'] = label
        else:
            raise ValueError('Dataset must be pandas.DataFrame or d8.image_classification.Dataset')

def _get_supported_models():
    all_models = get_model_list()
    blacklist = ['ssd', 'faster_rcnn', 'mask_rcnn', 'fcn', 'deeplab',
                 'psp', 'icnet', 'fastscnn', 'danet', 'yolo', 'pose',
                 'center_net', 'siamrpn', 'monodepth',
                 'ucf101', 'kinetics', 'voc', 'coco', 'citys', 'mhpv1',
                 'ade', 'hmdb51', 'sthsth', 'otb']
    cls_models = [m for m in all_models if not any(x in m for x in blacklist)]
    return cls_models

_SUPPORTED_MODELS = _get_supported_models()
