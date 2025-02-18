import json


with open("tcbSignChain.json") as f:
    json_input = f.read()

data = json.loads(json_input)
issuer_chain = data['SGX-TCB-Info-Issuer-Chain'].replace("%20", " ").replace("%0A", "\n").replace("%2B", "+").replace("%2F", "/").replace("%3D", "=")


with open('../quote_verifier/tcbSignChain.pem', 'w') as f:
    f.write(issuer_chain)

print("Certificates saved as 'pck_cert.pem' and 'issuer_chain.pem'.")
