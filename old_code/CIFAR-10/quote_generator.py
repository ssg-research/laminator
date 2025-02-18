
def generate_quote(user_data) -> bytes:
    with open('/dev/attestation/user_report_data', 'wb') as f:
        f.write(user_data)
    with open('/dev/attestation/quote', 'rb') as f:
        quote = f.read()
    return quote