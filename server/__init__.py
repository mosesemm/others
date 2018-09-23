import os
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

app_config = os.getenv('APP_CONFIG', 'server.config.DevelopmentConfig')
app.config.from_object(app_config)

bcrypt = Bcrypt(app)

from server.mongo_conn_setup import MongoConnectionSetup
mongo_db = MongoConnectionSetup()

from server.auth.views import auth_blueprint
app.register_blueprint(auth_blueprint)