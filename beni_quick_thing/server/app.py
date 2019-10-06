from flask import Flask, jsonify, request, session
from flask_session import Session
app = Flask(__name__, static_folder="../web", static_url_path="")
sess = Session()

app.secret_key = 'super secret key'
app.config["SESSION_TYPE"] = "filesystem"

sess.init_app(app)


@app.route("/")
def root():
    return app.send_static_file("index.html")

@app.route("/api/login", methods=['POST'])
def login():
    #store user information maybe in file
    data = request.json
    print(data)
    session['username'] = data["username"]

    return jsonify({"message": "started"}), 200


@app.route("/api/play/<video_id>", methods=['POST'])
def play(video_id):
    print("tesfting: {} ".format(request.json))
    print(session['username'])
    #start whatever stuff while the video playing
    return jsonify({"message": "done"}), 200

@app.route("/api/submit-survey/<user_id>", methods=['POST'])
def submitSurvey(user_id):
    print("tesfting: {} ".format(request.get_json(silent=True)))
    #store this somewhere, maybe in file
    return jsonify({"message": "done"}), 200

@app.route("/api/results/<user_id>", methods=['GET'])
def getResults(user_id):
    print("tesfting: {} ".format(user_id))
    #get results for both survey and system

    resulst = {
        "system": "whatever",
        'survey': "bad",
    }
    return jsonify(resulst), 200


if __name__ == "__main__":
    app.run(debug=True, port=3000)