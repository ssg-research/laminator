import argparse
import measured_file_read
import quote_generator
from pathlib import Path
import hashlib
import json
import base64
import torch
import time

from utils import (
    data_load_train,
    initialize_model,
    train_classifier,
)

def save_model_with_hashing(model, path):
    hasher = hashlib.sha512()
    # Initialize your MeasuredBytesIOWrite object with the hasher
    measured_file, file_object = measured_file_read.open_measured_write(path, "wb", hasher)
    
    # Save the model's state_dict using torch.save to the measured_file
    # Note: We directly use the file_object which is the actual file opened in binary write mode
    torch.save(model, file_object)
    
    # You can now access the hash of the data written to the file for verification or other purposes
    data_hash = hasher.digest()
    
    return data_hash

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
    parser.add_argument(
        '--name',
        type = str, 
        default ='census_model', 
        help='path of the model')                     

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    preprocess_start = time.time()
    data = data_load_train(Path('./data/census'))
    train_loader = torch.utils.data.DataLoader(dataset=data.train_set, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    name= args.name
    target_model_filename = f'./{name}.pt'
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    print("Time to load: ", preprocess_time)
    computate_start= time.time()
    target_model = initialize_model().to(args.device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    epochs=10
    target_model = train_classifier(target_model, train_loader,criterion=criterion, optimizer=optimizer, epochs=epochs, device=args.device)
    model_hash=save_model_with_hashing(target_model, target_model_filename)
    computate_end= time.time()
    computate_time= computate_end-computate_start
    model_hash_b64 = base64.b64encode(model_hash).decode('utf-8')
    manifest_file_hash = base64.b64encode(data.file.hasher.digest()).decode('utf-8')
    print("Time to computate: ", computate_time)
    computate_end= time.time()
    computate_time= computate_end-computate_start
    jsondata={}
    payload= {
     'name': model_hash_b64,
     'dataset-hash': manifest_file_hash,
     'training-parameters': {
         'epochs': epochs,
         'optimizer': 'Adam',
         'Scaler': base64.b64encode(data.scaler_hash).decode('utf-8'),
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