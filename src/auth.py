from datetime import datetime, timedelta

import jwt
from flask import Flask, jsonify, make_response, request, session

app = Flask(__name__)
app.config['SECRET_KEY'] = "63d3a56a02274adfa13333ec861fff32"

#  A. 인증 api
# POST method
# request: email, password
# response: auth key

@app.route("/auth", methods=["POST"])
def auth():
    if request.form['email'] == "abc@gmail.com" and request.form['password'] == "123123123":
        session['logged_in'] = True
        token = jwt.encode({
            'email': request.form['email'],
            'expiration': str(datetime.utcnow() + timedelta(minutes=30))
        }, app.config['SECRET_KEY'])
        return jsonify({'auth_key': token})
    else:
        return make_response("인증 실패", 403)

if __name__ == "__main__":
    app.run()
