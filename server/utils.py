import jwt
from datetime import datetime
from datetime import timedelta

from server import app

def encode_auth_token(user_id):
    try:
        payload = {
            'exp': datetime.utcnow() + timedelta(days=0, minutes=2),
            'iat': datetime.utcnow(),
            'sub': user_id
        }

        return jwt.encode(
            payload,
            app.config.get('SECRET_KEY'),
            algorithm= 'HS256'
        )
    except Exception as e:
        return e

def decode_auth_token(auth_token):
    try:
        payload = jwt.decode(auth_token, app.config.get('SECRET_KEY'))
        return payload
    except jwt.ExpiredSignatureError:
        return 'Signature expired. Please log in again'
    except jwt.InvalidTokenError:
        return 'Invalid token. Please log in again'