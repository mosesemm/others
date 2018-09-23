
import unittest
from server.utils import encode_auth_token, decode_auth_token
from tests.base import BaseTestCase

class TestUtils(BaseTestCase):

    def test_encode_auth_token(self):
        auth_token = encode_auth_token(232323)
        self.assertTrue(isinstance(auth_token, bytes))

    def test_decode_auth_token(self):
        auth_token = encode_auth_token(232323)
        self.assertTrue(decode_auth_token(auth_token)['sub'])

if __name__ == '__main__':
    unittest.main()
