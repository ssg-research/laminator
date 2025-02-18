import torch
import argparse
import utils
import pandas as pd
import measured_file_read
import hashlib
import quote_generator
import base64
import json
import time

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../', help='Root directory of models and datasets.')
    parser.add_argument('--dataset', type=str, default='census')
    parser.add_argument('--device', type=str, default=torch.device('cpu'), help='Device on which to run PyTorch')    
    return parser.parse_args()



def main(args: argparse.Namespace) -> None:
    preprocess_start = time.time()
    # Assuming the model and a single input instance are ready
    csv_path = "./input/input.csv"
    file= measured_file_read.open_measured(csv_path, 'rb')
    input_df = pd.read_csv(file, dtype=dtypes)
    scaler = utils.load_scaler("trained_scaler.pkl")
    model = measured_file_read.open_measured("census_model.pt", "rb")
    target_model = torch.load(model).to(args.device)
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    print("Time to load: ", preprocess_time)
    computate_start= time.time()
    prediction = utils.predict_input(target_model, input_df, args.device, scaler=scaler[0])
    model_hash_b64 = base64.b64encode(model.hasher.digest()).decode('utf-8')
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    jsondata={}
    payload={
        "name": model_hash_b64,
        "results": {
            "task": "image-recognition",
            "metrics": {
                'type': 'inference',
                'input': base64.b64encode(file.hasher.digest()).decode('utf-8'),
                'value': prediction,
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
    out_file=open('output_io.json', 'w')
    json.dump(jsondata, out_file)

if __name__ == '__main__':
    start=time.time()
    print("Start Time:", start)
    args = parse_args()
    main(args)
    end=time.time()
    duration = end-start
    print("Duration: ", duration)