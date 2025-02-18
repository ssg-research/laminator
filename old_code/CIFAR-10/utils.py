
import pickle
import os
import numpy as np
import hashlib
import measured_file_read

def unpickle(file)-> tuple:
    hasher = hashlib.sha512()
    
    # Use open_measured instead of open
    with measured_file_read.open_measured(file, 'rb', hasher) as fo:
        data = pickle.load(fo, encoding='bytes')
        hash = hasher.hexdigest()
    dict = (data, hash)
    return dict

def getset(label, data):
    data_set_images = []
    data_set_labels = []
    for y, x in zip(label, data):
        X_r = np.reshape(x[:1024], (32, 32))
        X_g = np.reshape(x[1024:2048], (32, 32))
        X_b = np.reshape(x[2048:], (32, 32))  # Splitting the RGB elements
        X = np.stack((X_r, X_g, X_b), axis=-1)  # Stacking R, G, B in 3-D
        data_set_images.append(X)
        data_set_labels.append(y)
    return np.array(data_set_images), np.array(data_set_labels)

def load_data_from_directory(path):
    data_values = []
    hashes = []
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            unpickled_data, data_hash = unpickle(file=full_path)
            data_values.append(unpickled_data)
            hashes.append(data_hash)
    return data_values, hashes

def preprocess_data(data) -> tuple:
    training_images = []
    training_labels = []
    for batch_data in data:
        # Assuming the first element of each tuple is the data dictionary
        labels = batch_data[b'labels']
        batch_images = batch_data[b'data']
        images, labels = getset(labels, batch_images)
        training_images.append(images)
        training_labels.append(labels)

    images = np.concatenate(training_images, axis=0)
    labels = np.concatenate(training_labels, axis=0)
    return images, labels
#    name = "cifar-10-batches-py/test_batch"
#    test = unpickle(name)
#    test_images, test_labels = getset(test[labels_dict[1]], test[labels_dict[2]])
#    print("Number of pictures in test set:", len(test_images))

def verify_manifest(manifest_hash, path):
    hasher = hashlib.sha512()
    
    with measured_file_read.open_measured(path, 'rb', hasher) as measured_file:
        content = measured_file.read()  # This read operation updates the hasher with the file's content.

    computed_hash = hasher.hexdigest()
    if(manifest_hash == computed_hash):
        return (set(content.decode('utf-8').strip().split('\n')), hasher.digest())
    else:
        exit()

def generate_manifest(path):
    hasher = hashlib.sha512()
    
    measured_file = measured_file_read.open_measured(path, 'rb', hasher)
    measured_file.read()
    measured_file.close()
    computed_hash = hasher.hexdigest()
    print(computed_hash)

def verify_hashes_against_manifest(expected_hashes, collected_hashes):
    verified_hashes = set(collected_hashes)  # Convert list to set for efficient lookup
    unmatched_hashes = verified_hashes - expected_hashes
    missing_hashes = expected_hashes - verified_hashes
    if not unmatched_hashes and not missing_hashes:
        print("All files verified successfully against the manifest.")
    else:
        if unmatched_hashes:
            print("The following file hashes did not match any expected hash:", unmatched_hashes)
        if missing_hashes:
            print("The following expected hashes were not found in the files:", missing_hashes)


def build_json_payload( name, results):
    template = {
        'name': name,
        'results': results
    }

    return template

