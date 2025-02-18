import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import model
import time
import measured_file_read
import argparse
import quote_generator
import hashlib
import json
import base64
import utils


def save_model_with_hashing(model, path):
    hasher = hashlib.sha512()
    # Initialize your MeasuredBytesIOWrite object with the hasher
    measured_file, file_object = measured_file_read.open_measured_write(path, "wb", hasher)
    
    # Save the model's state_dict using torch.save to the measured_file
    # Note: We directly use the file_object which is the actual file opened in binary write mode
    torch.save(model.state_dict(), file_object)
    
    # You can now access the hash of the data written to the file for verification or other purposes
    data_hash = measured_file.hasher.digest()
    
    return data_hash



def parse_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        type = str, 
        default ='cifar10_model', 
        help='path of the model')
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    preprocess_start = time.time()
    dataset_hash = "76f82a481e6b92371da8e2fddc88b7bf2561ac76753818afb646d494cd2bca87d13e62f6858812ebe5ff33ce7ca3c7317fccd0ab8df5473ed71c04d5539cd19e"
    manifest_file = utils.verify_manifest(
        dataset_hash,
        "manifest-train.dat")
    
    data_values, hashes = utils.load_data_from_directory("./train")
    utils.verify_hashes_against_manifest(manifest_file[0], hashes)
    training_images, training_labels = utils.preprocess_data(data=data_values)

    # Preprocess and convert to TensorDatasets and DataLoaders
    train_X = torch.tensor(training_images.transpose(0, 3, 1, 2), dtype=torch.float32)  # Reorder to NCHW format
    train_y = torch.tensor(training_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_X, train_y)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    print("Time to load: ", preprocess_time)
    computate_start= time.time()

    # Model, loss function, and optimizer
    net = model.Model()
    net.double()  # Your model uses double precision in forward, so ensure the data type matches
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training 
    epochs = 52
    for epoch in range(epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.double(), labels  # Match the model's expected data type

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    # Save the trained model
    name= args.name
    PATH = f'./{name}.pth'
    model_hash = save_model_with_hashing(net, PATH)
    model_hash_b64 = base64.b64encode(model_hash).decode('utf-8')
    manifest_file_hash = base64.b64encode(manifest_file[1]).decode('utf-8')
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    jsondata={}
    payload= {
     'name': model_hash_b64,
     'dataset-hash': manifest_file_hash,
     'training-parameters': {
         'epochs': epochs,
     }}
    quote_start = time.time()
    hasher= hashlib.sha512()
    payload_bytes = json.dumps(payload).encode('utf-8')
    hasher.update(payload_bytes)
    payload_hash = hasher.digest()
    quote=quote_generator.generate_quote(user_data=payload_hash)
    quote_end = time.time()
    jsondata['sgx-quote'] = base64.b64encode(quote).decode('utf-8')
    print("Time to form quote:", quote_end - quote_start)
    jsondata['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
    out_file=open('output_train.json', 'w')
    json.dump(jsondata, out_file)

if __name__ == '__main__':
    start=time.time()
    print("Start Time:", start)
    args = parse_args()
    main(args)
    end=time.time()
    duration = end-start
    print("Duration: ", duration)