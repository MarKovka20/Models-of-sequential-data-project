from utils import MetaParent

import torch
from sklearn.metrics import precision_score, accuracy_score


class BaseMetric(metaclass=MetaParent):
    pass


class StaticMetric(BaseMetric, config_name='dummy'):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric(BaseMetric, config_name='composite'):

    def __init__(self, metrics):
        self._metrics = metrics

    @classmethod
    def create_from_config(cls, config):
        return cls(metrics=[
            BaseMetric.create_from_config(cfg)
            for cfg in config['metrics']
        ])

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class PrecisionMetric(BaseMetric, config_name='precision'):

    def __call__(self, inputs):
        predicts = inputs['predicts']  # (batch_size)

        predicts = (predicts >= 0).long()  # (batch_size)
        labels = inputs['labels'].long()  # (batch_size)

        return accuracy_score(labels, predicts)  # TODO fix
