"""SpecializedModel

    The package contains all the AbstractModel implementations available in the framework.
    All the models here represent actual machine learning models, ready to be configured and trained.
"""

# Package level imports
from .modelTypes import DEEP_LEARNING_MODEL
from .deepLearningModel import DeepLearningModel

from .modelTypes import RANDOM_FOREST_MODEL
from .randomForestModel import RandomForestModel

from .modelTypes import SVM_MODEL
from .svmModel import SvmModel
# more models to be added
