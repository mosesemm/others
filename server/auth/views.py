
from flask import Blueprint, request, make_response, jsonify
from flask.views import MethodView

from server import bcrypt, mongo_db
from server.models import UserService
from server.utils import encode_auth_token, decode_auth_token

auth_blueprint = Blueprint('auth', __name__)

class RegisterAPI(MethodView):

    userService = UserService() 

    def post(self):
        post_data = request.get_json()
        user = self.userService.get_user_by_email(post_data['email'])
        print(user)
        if not user:
            try:
                user = dict(email=post_data.get('email'), 
                password=post_data.get('password'))

                id = self.userService.insert_user(user)
                auth_token = encode_auth_token(id)
                response_object = {
                    'status': 'success',
                    'message': 'Successfully registered.',
                    'auth_token': auth_token.decode()
                }

                return make_response(jsonify(response_object)), 201

            except Exception as e:
                response_object = {
                    'status': 'fail',
                    'message': 'Unexpected error occurred. Please try again.'
                }
                return make_response(jsonify(response_object)), 401
        else:
            response_object = {
                'status': 'fail',
                'message': 'User already exists. Please log in.'
            }
            return make_response(jsonify(response_object)), 202


registration_view = RegisterAPI.as_view('register_api')
auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST']
)
