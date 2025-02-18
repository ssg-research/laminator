from pathlib import Path

import torch
import measured_file_read
import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib

from model import BinaryNet

columns_after_encoding = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Unknown', 'workclass_Without-pay', 'education_11th', 'education_12th', 'education_1st-4th', 'education_5th-6th', 'education_7th-8th', 'education_9th', 'education_Assoc-acdm', 'education_Assoc-voc', 'education_Bachelors', 'education_Doctorate', 'education_HS-grad', 'education_Masters', 'education_Preschool', 'education_Prof-school', 'education_Some-college', 'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent', 'marital-status_Never-married', 'marital-status_Separated', 'marital-status_Widowed', 'occupation_Armed-Forces', 'occupation_Craft-repair', 'occupation_Exec-managerial', 'occupation_Farming-fishing', 'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct', 'occupation_Other-service', 'occupation_Priv-house-serv', 'occupation_Prof-specialty', 'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support', 'occupation_Transport-moving', 'occupation_Unknown', 'relationship_Not-in-family', 'relationship_Other-relative', 'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Unknown', 'native-country_Vietnam', 'native-country_Yugoslavia']

def load_census(path:  Path = Path('./data/census')
) -> Bunch:
    """
    Loads the Census Income dataset from https://archive.ics.uci.edu/dataset/20/census+income.
    Applies data standard data cleaning and one-hot encoding. Separates the sensitive attributes
    from the training and testing data.

    Args:
        path: str or Path object
            String or Path object indicating where to store the dataset.
        random_seed: int
            Determines random number generation for dataset shuffling. Pass an int
            for reproducible output across multiple function calls.
        return_x_y_z: bool
            If True, instead of returning a Bunch, it returns a tuple
    Returns:
        A Tuple with MeasuredFile object of the loaded data and a Dictionary-like object (:class:`~sklearn.utils.Bunch`), with the following attributes:
            train_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                training PyTorch models.

            test_set: :class:`~torch.utils.data.TensorDataset`
                A dataset of images and labels used to build a DataLoader for
                test PyTorch models.

        (x_train, x_test, y_train, y_test, z_train, z_test): tuple of ndarrays if return_x_z_y is true 
            ndarrays contain the data, the targets, and the sensitive attributes, each with a train/test split. 
    """ 
    dtypes = {
        'age': int, 
        'workclass': str, 
        'fnlwgt': int, 
        'education': str, 
        'education-num': int,
        'marital-status': str, 
        'occupation': str, 
        'relationship': str, 
        'race': str, 
        'sex': str,
        'capital-gain': int, 
        'capital-loss': int, 
        'hours-per-week': int, 
        'native-country': str, 
        'income': str
    }
    filename = path / 'adult.csv'
    file= measured_file_read.open_measured(str(filename), 'rb')
    adult_data = pd.read_csv(file, dtype=dtypes)

    # Split data into features / sensitive features / target
    sensitive_attributes = ['race', 'sex']
    sensitive_features = (adult_data.loc[:, sensitive_attributes]
                          .assign(race=lambda df: (df['race'] == 'White')
                                  .astype(int), sex=lambda df: (df['sex'] == 'Male')
                                  .astype(int)))
    target = (adult_data['income'] == '>50K').astype(int)
    to_drop = ['income', 'fnlwgt'] + sensitive_attributes
    features = (adult_data.drop(columns=to_drop).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
     # Split data into train / test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(features,
                                                              target,
                                                              sensitive_features,
                                                              test_size = 0.2,
                                                              random_state = 7)
    # Normalize data
    scaler= load_scaler("trained_scaler.pkl")
    def scale_df(df, scaler):
        return pd.DataFrame(scaler.transform(df),columns=df.columns, index=df.index)
    x_train = x_train.pipe(scale_df, scaler[0])
    x_test = x_test.pipe(scale_df, scaler[0])
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(x_train)).type(torch.FloatTensor),
                                               torch.from_numpy(np.array(y_train)).type(torch.LongTensor))
    
    test_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(x_test)).type(torch.FloatTensor),
                                              torch.from_numpy(np.array(y_test)).type(torch.LongTensor))
    
    return Bunch(
        file=file,
        test_set=test_set,
        train_set=train_set,
        scaler_hash=scaler[1],
    )
def load_scaler(path) -> tuple:
    file=measured_file_read.open_measured(path, 'rb')
    scaler=pickle.load(file)
    return (scaler, file.hasher.digest())

def load_data(dataset: str) -> Bunch:
    """
    Loads data given the dataset and the training size.
    
    Args:
        root: :class:~`pathlib.Path` or str
            Root directory of pipeline
    """
    data = load_census(Path('./data/census'))
    print(f'Dataset loaded: {dataset}')

    return data

def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 device: str,
                 sens_attr: bool = False
) -> float:
    """
    Calculates the classification accuracy of a model. 

    Args:
        model: :class:`~torch.nn.Module`
            The model to evaluate.
        data_loader: :class:'~torch.utils.data.DataLoader
            Input data to the model.
        device: str
            Device used for inference. Example: "cuda:0".

    Returns:
        The accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for tuple in data_loader:
            if sens_attr:
                x, y, _ = tuple[0].to(device), tuple[1].to(device), tuple[2].to(device)
            else:
                x, y = tuple[0].to(device), tuple[1].to(device)
            outputs = model(x)
            _, predictions = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predictions == y).sum().item()

    return 100 * correct / total

def predict_input(model, input_df, device, scaler):
    """
    Predicts the output for a single input DataFrame using the provided model.

    Args:
        model: The trained PyTorch model for prediction.
        input_df: DataFrame containing the input data.
        device: The computation device ('cuda' or 'cpu').
        scaler: The StandardScaler instance used to normalize the input data.   # Use MeasuredBytesIOWrite for writing
    with measured_file_read.MeasuredBytesIOWrite(hasher) as measured_writer:
        measured_writer.write(serialized_object)
        
        # Now, write the serialized object to a file
        with open(filepath, "wb") as file_handle:
            file_handle.write(measured_writer.getvalue())

    Returns:
        The predicted class for the input data.
    """
    model.eval()
    preprocessed_input = preprocess_single_input(
        input_df, 
        scaler=scaler,  # Assuming your scaler is loaded here as the first element of a tuple
        columns_after_encoding=columns_after_encoding,  # You need to provide this based on your training data
        sensitive_attributes=['race', 'sex'],
        to_drop=['income', 'fnlwgt', 'race', 'sex']
    )
    input_tensor = torch.tensor(preprocessed_input, dtype=torch.float)

    # Now you can use the `.to(device)` method, since `input_tensor` is a PyTorch tensor
    input_tensor = input_tensor.to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform the prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_classes = torch.max(output, 1)
        prediction = predicted_classes.sum().item()

    return prediction

def preprocess_single_input(input_df, scaler, columns_after_encoding, sensitive_attributes, to_drop):
    """
    Preprocess a single input dictionary for prediction, matching the training data preprocessing.

    Args:
        input_dict (dict): Input data as a dictionary.
        scaler (StandardScaler): The scaler used for normalizing training data.
        columns_after_encoding (list): List of columns after one-hot encoding and preprocessing.
        sensitive_attributes (list): List of sensitive attributes to handle separately or exclude.
        to_drop (list): List of columns to drop before encoding and scaling.

    Returns:
        torch.Tensor: A tensor representing the preprocessed and normalized input.
    """
    processed_df = input_df.drop(columns=to_drop)

    # Apply one-hot encoding
    processed_df = pd.get_dummies(processed_df)

    # Ensure all expected columns are present, fill missing with 0
    for col in columns_after_encoding:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Reorder columns to match training data
    processed_df = processed_df.reindex(columns=columns_after_encoding, fill_value=0)

    # Normalize using the loaded scaler
    normalized_features = scaler.transform(processed_df)
    
    return normalized_features

def data_load_train(path: Path = Path('./data/census')
) -> Bunch:
    
    dtypes = {
        'age': int, 
        'workclass': str, 
        'fnlwgt': int, 
        'education': str, 
        'education-num': int,
        'marital-status': str, 
        'occupation': str, 
        'relationship': str, 
        'race': str, 
        'sex': str,
        'capital-gain': int, 
        'capital-loss': int, 
        'hours-per-week': int, 
        'native-country': str, 
        'income': str
    }


    filename = path / 'adult.csv'
    file= measured_file_read.open_measured(str(filename), 'rb')
    adult_data = pd.read_csv(file, dtype=dtypes)

    # Split data into features / sensitive features / target
    sensitive_attributes = ['race', 'sex']
    sensitive_features = (adult_data.loc[:, sensitive_attributes]
                          .assign(race=lambda df: (df['race'] == 'White')
                                  .astype(int), sex=lambda df: (df['sex'] == 'Male')
                                  .astype(int)))
    target = (adult_data['income'] == '>50K').astype(int)
    to_drop = ['income', 'fnlwgt'] + sensitive_attributes
    features = (adult_data.drop(columns=to_drop).fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
     # Split data into train / test
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(features,
                                                              target,
                                                              sensitive_features,
                                                              test_size = 0.2,
                                                              random_state = 7)
    scaler = StandardScaler().fit(x_train)
    def scale_df(df, scaler):
        return pd.DataFrame(scaler.transform(df),columns=df.columns, index=df.index)
    x_train = x_train.pipe(scale_df, scaler)
    scaler_hash=save_scaler(scaler=scaler)
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(x_train)).type(torch.FloatTensor),
                                            torch.from_numpy(np.array(y_train)).type(torch.LongTensor))

    return Bunch(
        file=file,
        train_set=train_set,
        scaler_hash=scaler_hash
    )

def save_scaler(scaler):
    hasher = hashlib.sha512()
    # Serialize the object to bytes
    serialized_object = pickle.dumps(scaler)
    measured_writer, file_handle = measured_file_read.open_measured_write("trained_scaler.pkl", 'wb', hasher)
    try:
        measured_writer.write(serialized_object)
        file_handle.write(measured_writer.getvalue())
    finally:
        file_handle.close()
        measured_writer.close()
    
    return hasher.digest()


capacity_map = {
    'm1':{
        'vgg': 'VGG11',
        'linearnet': [128, 256, 128],
        'binarynet': [32, 64, 32]
    },
}


def initialize_model() -> torch.nn.Module:
    """
    Creates a model using the configuration provided.

    Args:
        model: str 
            Which model to initialize.
        model_capacity: str
            Size of the model.
        dataset: str
            The dataset that will beimport torch used to train the model/
        log: :class:~`logging.Logger` or None
            Logging facility.

    Returns:
        Path to the created directory.
    """
    model = BinaryNet(num_features=93, hidden_layer_sizes=capacity_map["m1"]['binarynet'])


    return model

def train_classifier(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: str, 
        sens_attr: bool = False
) -> torch.nn.Module:
    """
    Trains a classifier. 

    Args:
        model: :class:`~torch.nn.Module`
            Model to be trained.
        data_loader: :class:'~torch.utils.data.DataLoader
            Data used to train the model.
        criterion: :class:`~torch.nn.Module`
            Loss function for training model.
        optimizer: :class:`~torch.optim.Optimizer` 
            Optimizer for training model.
        epochs: int
            Determines number of iterations over training data.
        device: str
            Device used to train model. Example: "cuda:0".

    Returns:
        Trained model of type :class:`~torch.nn.Module`.
    """
    model.train()

    for epoch in range(epochs):
        correct = 0
        total = 0
        for tuple in data_loader:
            if sens_attr:
                x, y, _ = tuple[0].to(device), tuple[1].to(device), tuple[2].to(device)
            else:
                x, y = tuple[0].to(device), tuple[1].to(device)
            optimizer.zero_grad()
            output = model(x)
            _, predictions = torch.max(output,1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            correct += predictions.eq(y).sum().item()
            total += len(y)

        print(f'Train Epoch: {epoch} Loss: {loss.item():.6f} Acc: {correct/total*100:.2f}')
    return model

def load_data_distribution():
    """
    Loads the Census Income dataset and processes it by encoding, splitting, and normalizing.
    
    Returns:
        Tuple containing:
        - x_train (numpy.ndarray): Training features.
        - x_test (numpy.ndarray): Testing features.
        - y_train (numpy.ndarray): Training labels.
        - y_test (pandas.Series): Testing labels.
        - z_train (pandas.DataFrame): Training sensitive attributes.
        - z_test (pandas.DataFrame): Testing sensitive attributes.
    """
    dtypes = {
        'age': int,
        'workclass': str,
        'fnlwgt': int,
        'education': str,
        'education-num': int,
        'marital-status': str,
        'occupation': str,
        'relationship': str,
        'race': str,
        'sex': str,
        'capital-gain': int,
        'capital-loss': int,
        'hours-per-week': int,
        'native-country': str,
        'income': str
    }

    filename = 'data/census/adult.csv'
    measured_file = measured_file_read.open_measured(filename, 'rb')
    adult_data = pd.read_csv(measured_file, dtype=dtypes)

    sensitive_attributes = ['race', 'sex']
    sensitive_features = adult_data[sensitive_attributes].copy()
    sensitive_features['race'] = (adult_data['race'] == 'White').astype(int)
    sensitive_features['sex'] = (adult_data['sex'] == 'Male').astype(int)

    target = (adult_data['income'] == '>50K').astype(int)
    to_drop = ['income', 'fnlwgt'] + sensitive_attributes
    features = pd.get_dummies(adult_data.drop(columns=to_drop), drop_first=True)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        features,
        target,
        sensitive_features,
        test_size=0.2,
        stratify=target,
        random_state=7
    )

    #scaler = StandardScaler().fit(x_train)
    #measured_scaler = save_scaler(scaler)
    ##x_train = scaler.transform(x_train)

    return x_train, y_train.values, z_train.values, measured_file
