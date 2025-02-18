import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import model
import time
import measured_file_read
import quote_generator
import hashlib
import json
import base64
import utils

def main() -> None:
    preprocess_start = time.time()
    model_path= './cifar10_model.pth'

    manifest_file = utils.verify_manifest(
        "9189d657bd4dd50b67442848b9c537399401c3153307b7fdd3051e7e28f0b74bfdb8fa29c1ea515206f5ebaeb01e8dd9b49b08ec66f4c265b6623827e93fa1d9",
        "manifest-test.dat")
    
    data_values, hashes = utils.load_data_from_directory("./test")
    utils.verify_hashes_against_manifest(manifest_file[0], hashes)
    net = model.Model()
    net.double()
    measured_bytes_model = measured_file_read.open_measured(model_path, "rb")
    net.load_state_dict(torch.load(measured_bytes_model))
    test_images, test_labels = utils.preprocess_data(data=data_values)
    test_X = torch.tensor(test_images.transpose(0, 3, 1, 2), dtype=torch.float32)  # Reorder to NCHW format
    test_y = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_X, test_y)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    net.eval()
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    print("Time to load: ", preprocess_time)
    computate_start= time.time()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.double(), labels  # Ensure data type matches model
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = round(100 * correct / total, 5)
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    model_hash_b64 = base64.b64encode(measured_bytes_model.hasher.digest()).decode('utf-8')
    computate_end= time.time()
    computate_time= computate_end-computate_start
    manifest_file_hash = manifest_file_hash = base64.b64encode(manifest_file[1]).decode('utf-8')
    print("Time to computate: ", computate_time)
    jsondata={}
    payload= {
        'name': model_hash_b64,
        'results': {
            'task': "image-recognition",
            'dataset': "CIFAR-10",
            'metrics': {
                'type': 'accuracy',
                'value': accuracy,
                'dataset-hash': manifest_file_hash,
            }
        }
    }
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
    out_file=open('output_accuracy.json', 'w')
    json.dump(jsondata, out_file)

if __name__ == '__main__':
    start=time.time()
    print("Start Time:", start)
    main()
    end=time.time()
    duration = end-start
    print("Duration: ", duration)