from sklearn.model_selection import train_test_split
from src.model.training_model_helpers.dataset_splitter.splitter_validator import _validate_input, _validate_output
from src.model.schemas.dataset_schema import DatasetSplit
from src.model.schemas.dataset_schema import Dataset

"""
What:
    Dataset splitter for training and test usage.

Responsibilities:
    Set the splitter configurations.
    Split the dataset into two categories, (X = Landmarks, y =  Labels)
    train X_train, y_train and test X_test, y_test.

Preconditions:
    dataset_schema.py:
        Store all output in the dataclass for easy access.

    Validator:
        Handle all validation of dataset splitter module.
        →splitter_validator.py

Input:
    #NOTE: X_y_dataset: DATACLASS (Dataset)
        →landmarks:
            X: np.ndarray.
        →labels:
            y: np.ndarray.

    config_test_size:
        Configuration value of the test_size.
    config_random_state:
        Configuration value of the random state.

Process:
    Unpack X and y from Dataset.

    Validate input: (X: np.ndarray, y: np.ndarray).

    Split the dataset into two categories
    Train:
        X_train
        y_train
    Test:
        X_test
        y_test
    
    Validate Output: (X_train, X_test, y_train, y_test)

Output:
    #NOTE: DatasetSplit: DATACLASS (DatasetSplit)
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Invariants:
    Input landmarks (X), and labels (y) must be np.ndarray.
    Output X_train, X_test, y_train, y_test must be np.ndarray.
    DOES NOT MODIFY the dataset.
"""

#NOTE: MAIN FUNCTION ↓
#Split the data set. | Return: X_train, X_test, y_train, y_test
#Return: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
def split_dataset(X_y_dataset:Dataset , test_size: float, random_state: int) -> DatasetSplit:
    
    #Unpack the X and y.
    X = X_y_dataset.X_landmarks
    y = X_y_dataset.y_labels
    
    #Validator.
    _validate_input(X, y, test_size, random_state)
    
    #Split the dataset into train and test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    
    #Validator.
    _validate_output(X_train, X_test, y_train, y_test)
    
    return DatasetSplit(X_train, X_test, y_train, y_test)