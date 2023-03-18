import base64
import cv2
import json
import numpy as np
import tempfile
from amazoncaptcha import AmazonCaptcha
import time
import os
from scapy.all import *
from io import BytesIO
import jwt
from jwcrypto import jwk
from cryptography.hazmat.primitives import serialization


def binary_to_number(binary):
    number = 0
    length = len(binary)
    for i in range(0, length):
        if(binary[i] == '1'):
            number += 2**(length-i-1)
    return number


def base64_padding(s):
    return s + '=' * (4 - len(s) % 4)


def cipher_solver(question):
    # calc time
    text = base64_padding(question)
    newtext = base64.b64decode(text)
    # 101000111010101001100111010010000111100001101001011011001010100
    newtext = newtext.decode()
    newtext = newtext.replace("(", "").replace(")", "")
    newtext = newtext.split(",")
    lefttext = newtext[0]
    righttext = newtext[1]
    rightnumber = binary_to_number(righttext)

    strings = []
    for i in range(0, len(lefttext), 7):
        strings.append(lefttext[i:i+7])

    ans = ""

    for i in strings:
        ascii = binary_to_number(i)
        if(ascii <= 90):
            ascii -= rightnumber
            if(ascii < 65):
                ascii += 26
        else:
            ascii -= rightnumber
            if(ascii < 97):
                ascii += 26
        ascii = chr(ascii)

        ans += ascii
    return ans


def captcha_solver(question):
    # calculate time to solve
    captcha_array = []
    captcha_array = np.array(question, dtype=np.uint8)
    captcha_bgr = cv2.cvtColor(captcha_array, cv2.COLOR_GRAY2BGR)
    temp_file = tempfile.NamedTemporaryFile(
        suffix='.png', mode='w+b', delete=False)
    cv2.imwrite(temp_file.name, captcha_bgr)
    captcha = AmazonCaptcha(temp_file.name)
    solution = captcha.solve()
    temp_file.close()
    os.unlink(temp_file.name)
    return solution


def pcap_solver(question):
    # calc time
    # Return solution
    base64_string = base64_padding(question)
    pcap_bytes = base64.b64decode(base64_string)

    pcap_packets = rdpcap(BytesIO(pcap_bytes))

    dns_packets = [
        packet["ICMP"].qd.qname.decode() for packet in pcap_packets if packet.haslayer("ICMP")]

    def get_subdomain(domain):
        parts = domain.split('.')
        return parts[0] if len(parts) > 1 else ''

    dns_packets.sort(key=get_subdomain)
    editedList = []
    for dns in dns_packets:
        splitedText = dns.split(".")
        base = splitedText[1]
        editedList.append(base64.b64decode(base64_padding(base)).decode())
    solution = ''.join(editedList)
    return solution


def server_solver(question):
    splitedToken = question.split('.')
    headers = splitedToken[0]
    payload = splitedToken[1]

    decoded_bytes = base64.b64decode(base64_padding(headers))
    headers = json.loads(decoded_bytes.decode('utf-8'))

    decoded_bytes = base64.b64decode(base64_padding(payload))
    payload = json.loads(decoded_bytes.decode('utf-8'))

    new_payload = {key: value for key, value in payload.items()}
    new_payload["admin"] = "true"

    with open("private_key.pem", "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None)
    with open("public_key.pem", "rb") as f:
        public_key = jwk.JWK.from_pem(f.read())
    public_key["kid"] = headers["kid"]
    header = {'alg': 'RS256',
              'kid': headers["kid"], 'jwk': public_key, 'typ': 'JWT'}
    jwt_token = jwt.encode(new_payload, private_key,
                           algorithm="RS256", headers=header)
    return jwt_token
