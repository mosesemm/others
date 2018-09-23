import os

basedir = os.path.abspath(os.path.dirname(__file__))
mongo_connection_string = os.getenv("MONGO_CONNECTION_STRING", "mongodb://localhost")
default_secret = "this is my default secret, random things... hmm"

class BaseConfig:
    SECRET_KEY = os.getenv('SECRET_KEY', default_secret)
    DEBUG = False
    BCRYPT_LOG_ROUNDS = 13
    mongo_connection_string = mongo_connection_string

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    BCRYPT_LOG_ROUNDS = 4
class ProductionConfig(BaseConfig):
    DEBUG = False


