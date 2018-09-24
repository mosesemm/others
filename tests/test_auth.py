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
    
    def test_registered_user_login(self):

        with self.client:
            resp_register = self.client.post(
                '/auth/register',
                data=json.dumps(
                    dict(
                        email="mosd@simplesolutions.co.za",
                        password='43434'
                    )
                ),
                content_type='application/json'
            )

            data_register = json.loads(resp_register.data.decode())
            self.assertTrue(data_register['status'] == 'success')
            self.assertTrue(
                data_register['message'] == 'Successfully registered.')
            self.assertTrue(data_register['auth_token'])
            self.assertTrue(resp_register.content_type == 'application/json')
            self.assertEqual(resp_register.status_code, 201)

            response = self.client.post(
                '/auth/login',
                data=json.dumps(dict(
                    email="mosd@simplesolutions.co.za",
                    password='43434'
                )),
                content_type='application/json'
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'success')
            self.assertTrue(data['message'] == 'Successfully logged in.')
            self.assertTrue(data['auth_token'])
            self.assertTrue(response.content_type == 'application/json')
            self.assertEqual(response.status_code, 200)

    def test_non_registered_user_login(self):
        """ Test for login of non-registered user """
        with self.client:
            response = self.client.post(
                '/auth/login',
                data=json.dumps(dict(
                    email='joe@gmail.com',
                    password='123456'
                )),
                content_type='application/json'
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'fail')
            self.assertTrue(data['message'] == 'User does not exist.')
            self.assertTrue(response.content_type == 'application/json')
            self.assertEqual(response.status_code, 404)

    def test_user_status(self):
        """ Test for user status """
        with self.client:
            resp_register = self.client.post(
                '/auth/register',
                data=json.dumps(dict(
                    email='joe@gmail.com',
                    password='123456'
                )),
                content_type='application/json'
            )
            response = self.client.get(
                '/auth/status',
                headers=dict(
                    Authorization='Bearer ' + json.loads(
                        resp_register.data.decode()
                    )['auth_token']
                )
            )
            data = json.loads(response.data.decode())
            self.assertTrue(data['status'] == 'success')
            self.assertTrue(data['data'] is not None)
            self.assertTrue(data['data']['email'] == 'joe@gmail.com')
            self.assertTrue(data['data']['role'] == 'student')
            self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()