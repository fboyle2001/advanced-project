from .algorithm_base import BaseCLAlgorithm

from .offline import OfflineTraining
from .finetuning import Finetuning
from .ewc import ElasticWeightConsolidation
from .gdumb import GDumb
from .rainbow_online import RainbowOnline
from .rainbow_online_exp import RainbowOnlineExperimental
from .l2p import LearningToPrompt
from .scr import SupervisedContrastiveReplay
from .der_pp import DarkExperiencePlusPlus
from .rainbow_ncm import RainbowOnlineNCM

from .trial import TrialAdaptedSCR