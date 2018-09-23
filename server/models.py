
from server import mongo_db

class UserService:

    db = mongo_db.database

    def get_user_by_email(self, email):
        return self.db.users.find_one({'email': email})

    def insert_user(self, user):
        return str(self.db.users.insert_one(user).inserted_id)

