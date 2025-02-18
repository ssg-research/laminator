import argparse
import pandas as pd
import time
import hashlib
import utils
import base64
import quote_generator
import json

from utils import (
    load_data_distribution
)

def distribution_attestation_z(z):
    df = pd.DataFrame()
    df['z'] = z.tolist()
    total_race = df['z'].value_counts()[0] + df['z'].value_counts()[1]
    print("ratio of non-white: {:.2f}".format(df['z'].value_counts()[0]/total_race))
    actual_property = round(df['z'].value_counts()[0]/total_race,2)
    return actual_property

    

def distribution_attestation_z_y(z, y_train):
    df = pd.DataFrame()
    df['z'] = z.tolist()
    df['y'] = y_train.tolist()
    z0_y0 = df.value_counts(["z","y"])[0][0]
    z0_y1 = df.value_counts(["z","y"])[0][1]
    z1_y0 = df.value_counts(["z","y"])[1][0]
    z1_y1 = df.value_counts(["z","y"])[1][1]
    z0_y0 = round(z0_y0/len(z.tolist()),2)
    z0_y1 = round(z0_y1/len(z.tolist()),2)
    z1_y0 = round(z1_y0/len(z.tolist()),2)
    z1_y1 = round(z1_y1/len(z.tolist()),2)
    return z0_y0, z0_y1, z1_y0, z1_y1



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--attestation',
        type = str, 
        default ='race_x', 
        help='Options: race_x, race_z-y, sex_x, sex_z-y.')
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    preprocess_start = time.time()
    x_train, y_train, z_train, measured_file  = load_data_distribution()
    # check distributional properties
    type_attestation = args.attestation
    preprocess_end = time.time()
    preprocess_time= preprocess_end -preprocess_start
    computate_start= time.time()
    print("Time to load: ", preprocess_time)


    if type_attestation == "race_x":
            result = distribution_attestation_z(z_train[:,0])
    elif type_attestation == "sex_x":
            result = distribution_attestation_z(z_train[:,1])
    elif type_attestation == "race_z-y":
            result = distribution_attestation_z_y(z_train[:,0], y_train)
    elif type_attestation == "sex_z-y":
            result = distribution_attestation_z_y(z_train[:,1], y_train)
    computate_end= time.time()
    computate_time= computate_end-computate_start
    print("Time to computate: ", computate_time)
    hasher= hashlib.sha512()
    combined_value = measured_file.hasher.digest() + str(result).encode()
    hasher.update(combined_value)
    combined_hash=hasher.digest()
    jsondata={}
    quote_start = time.time()
    first_quote=quote_generator.generate_quote(user_data=combined_hash)
    jsondata['sgx-quote'] = base64.b64encode(first_quote).decode('utf-8')
    quote_end = time.time()
    print("Time to form quote:", quote_end - quote_start)
    jsondata['payload'] = base64.b64encode(combined_value).decode('utf-8')
    out_file=open('output.json', 'w')
    json.dump(jsondata, out_file)

if __name__ == '__main__':
    start=time.time()
    print("Start Time:", start)
    args = parse_args()
    main(args)
    end=time.time()
    duration = end-start
    print("Duration: ", duration)