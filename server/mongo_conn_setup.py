from pymongo import MongoClient
import os
from server import app

class MongoConnectionSetup():

    def __init__(self):
        self.client = MongoClient(app.config.get('mongo_connection_string'))
        self.database = self.client['college-admission']