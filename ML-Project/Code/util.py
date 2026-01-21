import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col=-1, add_intercept=False, labeled=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    print(f"load dataset from {csv_path}")

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)
    
    header_length = 0
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
        header_length = len(headers)
    
    # Load features and labels
    label_col = label_col % header_length
    if labeled:
        x_cols = [i for i in range(header_length) if i != label_col]
        l_cols = [label_col]
    else:
        x_cols = [i for i in range(header_length)]
        l_cols = []
    inputs = np.loadtxt(csv_path, delimiter=',', usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', usecols=l_cols).astype(int)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def save_dataset(output_path, inputs, labels = None):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    concat_data = inputs
    if labels is not None:
        concat_data = np.vstack((inputs.T, labels)).T
    np.savetxt(output_path, concat_data, delimiter=',')

def save_labels(csv_path, labels):
    directory = os.path.dirname(csv_path)
    if not os.path.exists(directory):
        os.makedirs(directory)    
        
    result_str = [f"{i},{label}\n" for i,label in enumerate(labels)]
    with open(csv_path, 'w') as f:
        f.writelines(["Id,Label\n"] + result_str)


def offline_test(model, csv_path, ratio=0.1, shuffle=True, scale=False, method="standard"):
    """Test model on local dataset with labels.

    Args:
        model: Model with predict method.
        csv_path: Path to CSV file containing dataset.
        ratio: Ratio of data to use as test set.
        scale: Whether to scale the data before training
        method: Scaling choices {"standard", "minmax", "none"}
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    inputs, labels = load_dataset(csv_path, labeled=True)

    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, labels, test_size=ratio, random_state=42, shuffle=shuffle)
    
    if scale:
        scaler = None
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()      

        if scaler is not None:
            train_inputs = scaler.fit_transform(train_inputs).astype(np.float32, copy=False)
            test_inputs = scaler.transform(test_inputs).astype(np.float32, copy=False)

    model.fit(train_inputs, train_labels)

    pred_labels = model.predict(test_inputs)
    accuracy = np.sum(pred_labels == test_labels) / len(test_labels)
    print(f"local test accuracy: {accuracy*100:.2f}%")


def online_test(model, train_path, test_path, save_path):
    """Test model on dataset without labels and save results.

    Args:
        model: Model with predict method.
        csv_path: Path to CSV file containing dataset.
        save_path: Path to save prediction results.
    """
    train_inputs, train_labels = load_dataset(train_path, labeled=True)
    model.fit(train_inputs, train_labels)

    test_inputs, _ = load_dataset(test_path, labeled=False)
    pred_labels = model.predict(test_inputs)
    save_labels(save_path, pred_labels)
    print(f"predict {len(pred_labels)} labels and saved to {save_path}")

def online_tests(models_paths_map, train_path, test_path):
    """Test multiple models on dataset without labels and save results.

    Args:
        models_paths_map: A dict mapping model name to model path.
        train_path: Path to training CSV file containing dataset.
        test_path: Path to testing CSV file containing dataset.
    """
    train_inputs, train_labels = load_dataset(train_path, labeled=True)

    test_inputs, _ = load_dataset(test_path, labeled=False)

    for model, model_save_path in models_paths_map.items():
        print(f"Training with params: {model.get_params()}")
        model.fit(train_inputs, train_labels)
        pred_labels = model.predict(test_inputs)
        save_labels(model_save_path, pred_labels)
        print(f"[{model.__class__.__name__}] predict {len(pred_labels)} labels and saved to {model_save_path}")


def scale(method="standard", suffix="sampled"):
    """Scale the train and test dataset
    
    Args:
        method: Scaling choices {"standard", "minmax", "none"}
        suffix: scaled dataset file suffix
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    scaler = None
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()   
 
    train_path = "./data/train.csv"
    scaled_train_path = f"./data/train-{suffix}.csv"
    train_inputs, train_labels = load_dataset(train_path, labeled=True)

    if scaler is not None:
        train_inputs = scaler.fit_transform(train_inputs).astype(np.float32, copy=False)

    save_dataset(scaled_train_path, train_inputs, train_labels)

    test_path = "./data/test.csv"
    scaled_test_path = f"./data/test-{suffix}.csv"    
    test_inputs,_ = load_dataset(test_path, labeled=False)

    if scaler is not None:
        test_inputs = scaler.fit_transform(test_inputs).astype(np.float32, copy=False)

    save_dataset(scaled_test_path, test_inputs)

def save_model(model, model_path):
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)    
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


