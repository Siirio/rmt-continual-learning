"""Continual-learning strategies with compatible execution interfaces."""

from .base import EvaluationFn, StageResult, StrategyResult, TrainFn
from .experience_replay import ExperienceReplay, ReplayMemory
from .ewc import EWCStrategy, OnlineEWC, select_ewc_lambda
from .joint_training import JointTraining
from .sequential_lora import SequentialLoRA

__all__ = [
    "EvaluationFn",
    "ExperienceReplay",
    "EWCStrategy",
    "JointTraining",
    "OnlineEWC",
    "SequentialLoRA",
    "ReplayMemory",
    "StageResult",
    "StrategyResult",
    "TrainFn",
    "select_ewc_lambda",
]


