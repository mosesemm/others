from flask_testing import TestCase
from server import app, mongo_db

class BaseTestCase(TestCase):
    ''' Base Tests '''
    
    def create_app(self):
        app.config.from_object('server.config.DevelopmentConfig')
        return app

    def tearDown(self):
        mongo_db.database.command("dropDatabase")
