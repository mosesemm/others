
from server import mongo_db
from bson.objectid import ObjectId

class UserService:

    db = mongo_db.database

    def get_user_by_email(self, email):
        return self.db.users.find_one({'email': email})

    def get_user_by_id(self, id):
        return self.db.users.find_one({'_id': ObjectId(id)})

    def insert_user(self, user):
        return str(self.db.users.insert_one(user).inserted_id)

