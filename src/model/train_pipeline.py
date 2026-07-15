#dataset_loader.py
from src.model.training_model_helpers.dataset_loader.dataset_loader import load_dataset
#dataset_splitter.py
from src.model.training_model_helpers.dataset_splitter.dataset_splitter import split_dataset
#model_trainer.py
from src.model.training_model_helpers.model_trainer.model_trainer import model_trainer
#prediction_model.py
from src.model.training_model_helpers.predict_model.prediction_model import model_predictor
#model_accuracy.py
from src.model.training_model_helpers.model_accuracy.model_accuracy import get_accuracy_score
#save_model.py
from src.model.training_model_helpers.save_model.save_trained_model import save_trained_model
#model config schema.py
from src.model.schemas.config_schema import ModelConfig

from sklearn.base import ClassifierMixin
from pathlib import Path

class TrainModelPipeline:

    """
    What:
        Trains and save ML model for pose classification.

    Pipeline:
        dataset_loader.py
            ↓
        dataset_splitter.py
            ↓
        model_trainer.py
            ↓
        predict_model.py
            ↓
        model_accuracy.py 
            ↓
        save_model.py 

    Responbilities:
        Take the datasets from NPZ file.
        Train the model using the datasets.
        Save the trained model to "trained_models" folder

    Preconditions:
        mode_config_loader.py:
            The configuration loader of the model.

        dataset_loader.py: 
            Load the dataset from NPZ
            
        dataset_splitter.py: 
            Split the dataset in two categories: train and test
            
        model_trainer.py: 
            Train the model using the train datasets
            
        predict_model.py: 
            Get the trained model prediction
            
        model_accurary.py: 
            Evaluate the accuracy of the trained model
            
        save_model.py: 
            Save the trained model into folder.

    Input:
        model: ClassifierMixin
            The Chosen classifier model with specifications.

        config: ModelConfig
            The configuration of the model chosen.

    Process:
        1. Load features and labels.
        2. Split into training and testing sets.
        3. Initialize the ML algorithm.
        4. Train the model using fit().
        5. Evaluate performance on the test set.
        6. Save the trained model.

    Output:
        accuracy: float
            The accuracy of the model
        
        pose_classifier.joblib
    """

    """
    Variables:
        self.model
        self.accuracy
        self.config
    """

    def __init__(self, model: ClassifierMixin, config: ModelConfig) -> None:
            
            #Set the chosen model.
            self.model = model

            #Config.
            self.config = config
    
            #Store training result.
            self.trained_model: ClassifierMixin | None = None

            #Store model accuracy.
            self.accuracy: float | None = None

    """
    MAIN PROGRAM ↓
    """
    #Train and save the model. | Return: accuracy score (float).
    def train_and_save_model(self) -> float:

        """
        Pipeline:
            dataset_loader.py
                ↓
            dataset_splitter.py
                ↓
            model_trainer.py
                ↓
            predict_model.py
                ↓
            model_accuracy.py 
                ↓
            save_model.py 
        """
        
        #Load the dataset from NPZ.
        X_y_dataset = load_dataset(Path(self.config.dataset_file_path))

        #Split the datasets.
        splitted_dataset = split_dataset(X_y_dataset, self.config.test_size, self.config.random_state)

        #Train the model using datasets.
        self.trained_model = model_trainer(splitted_dataset, self.model)

        #Get the model predictions.
        predictions = model_predictor(splitted_dataset.X_test, self.trained_model)

        #Evaluate the model accuracy.
        self.accuracy = get_accuracy_score(predictions, splitted_dataset.y_test)

        #Save the trained model.
        save_trained_model(self.trained_model, Path(self.config.trained_models_path))

        return self.accuracy
