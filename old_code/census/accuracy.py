import argparse
import measured_file_read
import quote_generator
from pathlib import Path
import hashlib
import json
import base64
import time

import torch
from utils import (
    load_data, 
    get_accuracy
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', 
                        type = str, 
                        default = '../', 
                        help='Root directory of models and datasets.')
    parser.add_argument('--dataset', type = str, default = 'census')
    parser.add_argument('--device', 
                        type = str, 
                        default = torch.device('cpu'), 
                        help = 'Device on which to run PyTorch')  

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    preprocess_start = time.time()
    root_dir = Path(args.root)
    # Load dataset and create data loaders
    data = load_data(root_dir)
    test_loader = torch.utils.data.DataLoader(dataset=data.test_set, batch_size=32, shuffle=False)
    model = measured_file_read.open_measured("census_model.pt", "rb")
    target_model = torch.load(model)
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    print("Time to load: ", preprocess_time)
    computate_start= time.time()
    accuracy = get_accuracy(target_model, test_loader, args.device)       
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    model_hash = model.hasher.digest()
    model_hash_b64 = base64.b64encode(model_hash).decode('utf-8')
    manifest_file_hash = base64.b64encode(data.file.hasher.digest()).decode('utf-8')
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
    args = parse_args()
    main(args)
    end=time.time()
    duration = end-start
    print("Duration: ", duration)