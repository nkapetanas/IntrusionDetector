"""Default configuration for model architecture and training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class _Parameteres(object):
    "Wrapper class"
    pass


def model_config(vocab_size=30887,
                 dropout_rate=0.3,
                 time_steps=51,
                 lr=0.000001,
                 token_embedding_dim=40,
                 num_of_fields=41,
                 loss="binary_crossentropy"):
    """Creates a model configuration object.
    Args:
        vocab_size: size of vocabulary computed by preprocess_data.py
        dropout_rate: rate of dropout layers in model.
        lr: learning rate
        token_embedding_dim: dimension of token embeddings, computed by the embedding layer in the model
        num_of_fields: number of fields in the kdd logs.
        loss: loss function used during training.
    Returns:
        An object containing model configuration parameters.
    """
        
        
    config = _Parameteres()
    config.vocab_size = vocab_size
    config.dropout_rate = dropout_rate
    config.time_steps = time_steps
    config.lr = lr
    config.token_embedding_dim = token_embedding_dim
    config.num_of_fields = num_of_fields
    config.loss = loss
    return config


def train_config(input_directory_train="data_1/",
                 vocabulary_dir="vocabulary.pickle",
                 frequency_dict_dir="word_frequencies.pickle",
                 batch_size=128,
                 epochs=4,
                 train_examples=4653470,
                 validation_examples=244970,
                 checkpoint_path="kdd_models/"):
    """Creates a model configuration object.
    Args:
        input_directory_train: directory with files with training data and labels,
        vocabulary_dir : directory with vocabulary file (token: index)
        frequency_dict_dir: directory with token frequencies file (token:frequency of appearence)
        batch size: size of batch
        epochs: training epochs
        train examples: number of training instances
        validation examples: number of validation instances
        checkpoint_path: path of saving the model and its weights at the end of each epoch
    Returns:
        An object containing train configuration parameters.
    """
        
        
    config = _Parameteres()
    config.input_directory_train = input_directory_train
    config.vocabulary_dir = vocabulary_dir
    config.frequency_dict_dir = frequency_dict_dir
    config.batch_size = batch_size
    config.epochs = epochs
    config.train_examples = train_examples
    config.validation_examples = validation_examples
    config.model_path = checkpoint_path + "model_attention_weighted_crossentropy_keras.json"
    config.weights_path = checkpoint_path + "model_attention_weighted_crossentropy_keras.json"
    return config
 
 
    
def test_config(input_directory_test="data_test/",
                vocabulary_dir="vocabulary.pickle",
                frequency_dict_dir="word_frequencies.pickle",
                checkpoint_path="kdd_models/",
                batch_size=128,
                test_examples=50001*2):
    """Creates a model configuration object.
    Args:
        input_directory_test directory with files with test data and labels,
        vocabulary_dir : directory with vocabulary file (token: index)
        frequency_dict_dir: directory with token frequencies file (token:frequency of appearence)
        batch size: size of batch
        epochs: training epochs
        test examples: number of test instances
        checkpoint_path: path of saving the model and its weights at the end of each epoch
    Returns:
        An object containing test configuration parameters.
    """  
        
    config = _Parameteres()
    config.input_directory_test = input_directory_test
    config.vocabulary_dir = vocabulary_dir
    config.frequency_dict_dir = frequency_dict_dir
    config.batch_size = batch_size
    config.test_examples = test_examples
    config.model_path = checkpoint_path + "model_keras.json"
    config.weights_path = checkpoint_path + "model_keras.json"
    return config
