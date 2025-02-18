# Attestating ML properties using Intel SGX

First you need to setup your own DCAP setup for a computer:
https://www.intel.com/content/www/us/en/developer/articles/guide/intel-software-guard-extensions-data-center-attestation-primitives-quick-install-guide.html

and

https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/main/QuoteGeneration/pccs/README.md

Npm package "n" might help you to get a proper nodejs version to work: https://www.npmjs.com/package/n

Examples for CIFAR-10 and Census can be found under respective directories with their own docker files.
To setup docker environment configured to run examples, each docker file for the respective example can be found under the directories:

```
docker build -t {container_name} .
docker run --device /dev/sgx_enclave --device /dev/sgx_provision -it {container_name}
```
Note: if running on azure, uncomment these lines from Dockerfile. About Azure DCAP: https://learn.microsoft.com/en-us/azure/security/fundamentals/trusted-hardware-identity-management
```

# Download the .deb package
#RUN wget https://packages.microsoft.com/ubuntu/20.04/prod/pool/main/a/az-dcap-client/az-dcap-client_1.12.0_amd64.deb

# Install the .deb package
#RUN gdebi --non-interactive az-dcap-client_1.12.0_amd64.deb
```

To run the attestation generation:
```
make -f Makefile.{attestation} clean
make -f Makefile.{attestation} SGX=1
gramine-sgx ./{attestation} {attestation}.py
```
Note: {attestation} can be train/accuracy/io
Note: Run training first to get a model.
Note: If testing cifar-10, run the calculate_manifest_hash script first.

To run verifier under script/, porvide the output files from script to same directory and run
```
python3 verifier.py
```
Scripts folder container also contains useful tools for checking base64 valued string and parsing x509 certificates custom parts

To generate your own attested ML model instance, you have to keep track of few things:
- First make sure allowed files and trusted files (Info can be found here: https://gramine.readthedocs.io/en/stable/manifest-syntax.html) are setup correctly in your gramine manifest file. Trusted files are part of the trusted base and allowed files not. Every single allowed file e.g. input and output for inference need to bet hashed. measured_file_read contains helper functions for reading a file and hashing it. These hashes should be someway part of your user data passed to quote generator.
- Second, to generate quote, use the quote_generator.py generate_quote function. Pass your user data to this function and it will return bytes of a sgx quote in the manifest.
- Return quote to the user, remember to mark it as allowed file.

Generalization of a custom implementation:
```
#loading necessary allowed files
measured_file_read.open_measured(file1)
measured_file_read.open_measured(file2)
#computations etc. training model
#Forming user data.. Max size is 64B (size of SHA-512). remember to add hashes of input
user_data = .... file1 ... file2
#pass user data to quote generator
quote = quote_generator.generate_quote(user_data)
# pass quote to output
output = quote, payload
```



Useful links:

- https://gramine.readthedocs.io/en/stable/
- https://gramine.readthedocs.io/en/stable/manifest-syntax.html
- https://gramine.readthedocs.io/en/stable/attestation.html
- https://gramine.readthedocs.io/en/stable/run-sample-application.html
- https://github.com/intel/SGXDataCenterAttestationPrimitives
- https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_ECDSA_QuoteLibReference_DCAP_API.pdf
- https://learn.microsoft.com/en-us/azure/security/fundamentals/trusted-hardware-identity-management