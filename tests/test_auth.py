import unittest
import json 

from server import mongo_db
from tests.base import BaseTestCase
from server.models import UserService

class TestAuthBlueprint(BaseTestCase):
    
    userService = UserService()

    def test_registration(self):
        with self.client:
            response = self.client.post(
                '/auth/register',
                data = json.dumps(dict(email='test@simplesolutions.co.za',password='3232423')),
                content_type="application/json"

            )

            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'success')
            self.assertTrue(data['message'] == 'Successfully registered.')
            self.assertTrue(data['auth_token'])
            self.assertTrue(response.content_type == 'application/json')
            self.assertTrue(response.status_code, 201)
    
    def test_register_with_already_registered_user(self):
        self.userService.insert_user(dict(email='test@simplesolutions.co.za',password='3232423'))

        with self.client:
            response = self.client.post(
                '/auth/register',
                data=json.dumps(dict(email='test@simplesolutions.co.za',password='3232423')),
                content_type='application/json'
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'fail')
            self.assertTrue(data['message'] == 'User already exists. Please log in.')
            self.assertTrue(response.content_type == 'application/json')
            self.assertEqual(response.status_code, 202)

if __name__ == '__main__':
    unittest.main()