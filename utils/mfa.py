# utils/mfa.py — TOTP MFA helper

import pyotp, qrcode, io, base64
from config import MFA_ISSUER, MFA_WINDOW


def generate_secret():
    return pyotp.random_base32()


def get_qr_base64(secret, username):
    uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=MFA_ISSUER)
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def verify_code(secret, code):
    return pyotp.TOTP(secret).verify(code, valid_window=MFA_WINDOW)


def current_code(secret):
    return pyotp.TOTP(secret).now()
