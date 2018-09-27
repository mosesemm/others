
from flask import Blueprint, request, make_response, jsonify
from flask.views import MethodView

from server import bcrypt, mongo_db, app
from server.models import UserService
from server.utils import encode_auth_token, decode_auth_token
from datetime import datetime
from server.ocr_service import parse_certifcate
from io import BytesIO
from server.assessment_service import check_course_acceptence
from server.certificate_verification import verify_certificate

auth_blueprint = Blueprint('auth', __name__)


class ParseCertificateAPI(MethodView):

    def post(self):

        if 'file' not in request.files:
            response_object = {
                'status': 'fail',
                'message': 'File not attached'
            }
            return make_response(jsonify(response_object)), 400

        try:

            file = request.files['file']

            response_object = {
                'status': 'success',
                'data': parse_certifcate(BytesIO(file.read()))
            }
            return make_response(jsonify(response_object)), 201

        except Exception as e:
            print(e)
            response_object = {
                'status': 'fail',
                'message': 'Unexpected error occurred. Please try again.'
            }
            return make_response(jsonify(response_object)), 500

class CertVerifyAPI(MethodView):

    def post(self):
        post_data = request.get_json()
        
        try:

            results = verify_certificate(post_data['exam_number'], post_data['subjects'])

            response_object = {
                'status': 'success',
                'results': results,
            }

            return make_response(jsonify(response_object)), 201

        except Exception as e:
            print(e)
            response_object = {
                'status': 'fail',
                'message': 'Unexpected error occurred. Please try again.'
            }
            return make_response(jsonify(response_object)), 401

class AssessmentAPI(MethodView):

    def post(self):
        post_data = request.get_json()
        print(post_data)
        
        try:

            results_courses = check_course_acceptence(post_data['subjects'], post_data['course'])
            results_cert = verify_certificate(post_data['exam_number'], post_data['subjects'])

            response_object = {
                'status': 'success',
                'results_courses': results_courses,
                'results_cert': results_cert,
            }

            return make_response(jsonify(response_object)), 201

        except Exception as e:
            print(e)
            response_object = {
                'status': 'fail',
                'message': 'Unexpected error occurred. Please try again.'
            }
            return make_response(jsonify(response_object)), 400

class RegisterAPI(MethodView):

    userService = UserService() 

    def post(self):
        post_data = request.get_json()
        user = self.userService.get_user_by_email(post_data['email'])
        if not user:
            try:
                user = dict(email=post_data.get('email'), 
                password = bcrypt.generate_password_hash(post_data.get('password'), app.config.get('BCRYPT_LOG_ROUNDS')).decode(),
                role = 'student',
                registered_on = datetime.now())

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

class LoginAPI(MethodView):
    """
    User Login Resource
    """
    userService = UserService() 

    def post(self):
        # get the post data
        post_data = request.get_json()
        try:
            # fetch the user data
            user = self.userService.get_user_by_email(post_data['email'])
            print(user)
            if user and bcrypt.check_password_hash(user['password'], post_data.get('password')):

                auth_token = encode_auth_token(str(user['_id']))
                if auth_token:
                    responseObject = {
                        'status': 'success',
                        'message': 'Successfully logged in.',
                        'auth_token': auth_token.decode()
                    }
                    return make_response(jsonify(responseObject)), 200
            else:
                responseObject = {
                    'status': 'fail',
                    'message': 'User does not exist.'
                }
                return make_response(jsonify(responseObject)), 404
        except Exception as e:
            print(e)
            responseObject = {
                'status': 'fail',
                'message': 'Try again'
            }
            return make_response(jsonify(responseObject)), 500

class UserAPI(MethodView):
    """
    User Resource
    """

    userService = UserService() 

    def get(self):
        # get the auth token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            auth_token = auth_header.split(" ")[1]
        else:
            auth_token = ''
        if auth_token:
            resp = decode_auth_token(auth_token)
            if not isinstance(resp, str):
                print(resp['sub'])
                user = self.userService.get_user_by_id(resp['sub'])
                responseObject = {
                    'status': 'success',
                    'data': {
                        'user_id': str(user['_id']),
                        'email': user['email'],
                        'role': user['role'],
                        'registered_on': user['registered_on']
                    }
                }
                return make_response(jsonify(responseObject)), 200
            responseObject = {
                'status': 'fail',
                'message': resp
            }
            return make_response(jsonify(responseObject)), 401
        else:
            responseObject = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return make_response(jsonify(responseObject)), 401


registration_view = RegisterAPI.as_view('register_api')
login_view = LoginAPI.as_view('login_api')
user_view = UserAPI.as_view('user_api')
parse_cert_view = ParseCertificateAPI.as_view('parse_cert_view')
cert_verify_view = CertVerifyAPI.as_view('cert_verify_view')
assessment_view = AssessmentAPI.as_view('assessment_view')

auth_blueprint.add_url_rule(
    '/auth/register',
    view_func=registration_view,
    methods=['POST']
)
auth_blueprint.add_url_rule(
    '/auth/login',
    view_func=login_view,
    methods=['POST']
)

auth_blueprint.add_url_rule(
    '/auth/status',
    view_func=user_view,
    methods=['GET']
)

auth_blueprint.add_url_rule(
    '/auth/parse-certificate',
    view_func=parse_cert_view,
    methods=['POST']
)


auth_blueprint.add_url_rule(
    '/auth/cert-verify',
    view_func=cert_verify_view,
    methods=['POST']
)

auth_blueprint.add_url_rule(
    '/auth/assessment',
    view_func=assessment_view,
    methods=['POST']
)

