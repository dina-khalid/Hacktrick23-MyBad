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
    start = time.time()
    text = base64_padding(question)
    newtext = base64.b64decode(text)
    # 101000111010101001100111010010000111100001101001011011001010100
    newtext = str(newtext)
    newtext = newtext[3:-2]
    lefttext = newtext[0:-5]
    righttext = newtext[-4:]
    rightnumber = binary_to_number(righttext)
    # print(rightnumber)

    strings = []
    for i in range(0, 63, 7):
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
    end = time.time()
    print("Time to cipher: ", end - start)
    return ans


def captcha_solver(question):
    # calculate time to solve
    start_time = time.time()
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
    end_time = time.time()
    print("Time to captcha: ", end_time - start_time)
    return solution


def pcap_solver(question):
    # calc time
    start = time.time()
    # Return solution

    base64_string = base64_padding(question)
    pcap_bytes = base64.b64decode(base64_string)

    malicious_ip = '188.68.45.12'

    pcap_packets = rdpcap(BytesIO(pcap_bytes))

    dns_packets = [
        packet for packet in pcap_packets if packet.haslayer("ICMP")]

    list = []
    for packet in dns_packets:
        list.append(packet["ICMP"].qd.qname.decode())

    def get_subdomain(domain):
        parts = domain.split('.')
        return parts[0] if len(parts) > 1 else ''

    list = sorted(list, key=get_subdomain)
    editedList = []
    for dns in list:
        splitedText = dns.split(".")
        editedList.append(splitedText[1])
    end = time.time()
    print("Time to pcap: ", end - start)
    return base64.b64decode(''.join(editedList)).decode("ISO-8859-1")


def server_solver(question):
    splitedToken = question.split('.')
    headers = splitedToken[0]
    payload = splitedToken[1]

    decoded_bytes = base64.b64decode(base64_padding(headers))
    decoded_string = decoded_bytes.decode('utf-8')
    headers = json.loads(decoded_string)

    decoded_bytes = base64.b64decode(base64_padding(payload))
    decoded_string = decoded_bytes.decode('utf-8')
    payload = json.loads(decoded_string)

    new_payload = {
        **payload,
        "admin": True
    }

    private_key = jwk.JWK.generate(kty="RSA", size=2048, kid=headers["kid"])
    public_key = private_key.export(False)

    public_key = json.loads(public_key)

    header = {'alg': 'RS256', 'kid': headers["kid"], 'jwk': public_key}
    jwt_token = jwt.encode(new_payload, private_key.export_to_pem(
        private_key=True, password=None), algorithm="RS256", headers=header)
    return jwt_token
