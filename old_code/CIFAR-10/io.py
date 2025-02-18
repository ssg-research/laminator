import torch
import numpy as np
import model
import time
from PIL import Image
import measured_file_read
import quote_generator
import hashlib
import json
import base64
import utils

def main() -> None:
    load= time.time()
    image_path = "./input/last_image.png"
    measured_input = measured_file_read.open_measured(image_path, "rb")
    image = Image.open(measured_input).convert('RGB')
    end_load = time.time()
    load_time = end_load - load
    print("Time to load: ", load_time)
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array.transpose((2, 0, 1)), dtype=torch.float).unsqueeze(0)


    computate_start= time.time()
    net = model.Model()
    net.double()  # Ensure your model is in double precision mode
    model_path = './cifar10_model.pth'
    measured_bytes_model = measured_file_read.open_measured(model_path, "rb")
    net.load_state_dict(torch.load(measured_bytes_model))
    net.eval()
    # Make a prediction
    with torch.no_grad():
        image_tensor = image_tensor.double()  # Ensure the tensor type matches the model's configuration
        outputs = net(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to convert output to probabilities
        max_prob, predicted = torch.max(probabilities, 1)
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    model_hash_b64 = base64.b64encode(measured_bytes_model.hasher.digest()).decode('utf-8')
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
                'input': base64.b64encode(measured_input.hasher.digest()).decode('utf-8'),
                'value': predicted.item(),
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
    for i in range(100):
        start_run=time.time()
        main()
        end=time.time()
        duration = end-start_run
        print("Duration: ", duration)