## Common dependencies 
sudo apt-get install -y build-essential \
    autoconf bison gawk nasm ninja-build pkg-config python3 python3-click \
    python3-jinja2 python3-pip python3-pyelftools wget
sudo python3 -m pip install 'meson>=0.56' 'tomli>=1.1.0' 'tomli-w>=0.4.0'

## Dependencies for SGX
# Required Packages
sudo apt-get install -y libprotobuf-c-dev protobuf-c-compiler \
    protobuf-compiler python3-cryptography python3-pip python3-protobuf

# Install Intel SGX SDK/PSW
sudo curl -fsSLo /usr/share/keyrings/intel-sgx-deb.asc https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-sgx-deb.asc] https://download.01.org/intel-sgx/sgx_repo/ubuntu $(lsb_release -sc) main" | sudo tee /etc/apt/sources.list.d/intel-sgx.list
sudo apt-get update
sudo apt-get install libsgx-epid libsgx-quote-ex libsgx-dcap-ql libsgx-dcap-quote-verify-dev

## Build Gramine
git clone https://github.com/gramineproject/gramine.git
cd gramine/
meson setup build/ --buildtype=release -Ddirect=enabled -Dsgx=enabled -Ddcap=enabled
ninja -C build/ 
sudo ninja -C build/ install
gramine-sgx-gen-private-key

# # Build the Secret Provisioning Server Not required for our use case
# cd CI-Examples/ra-tls-secret-prov
# make app dcap RA_TYPE=dcap

## Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu