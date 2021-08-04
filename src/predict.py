import json
from functools import wraps
from json.decoder import JSONDecodeError

import jwt
from flask import Flask, jsonify, request
from numpy import argmax

from model_dependency import ModelDependency

categories = ["경영, 비즈니스", "개발", "디자인", "마케팅, 광고"]
cat_to_index = {cat: i for i, cat in enumerate(categories) }
index_to_cat = {i: cat for i, cat in enumerate(categories) }

max_len = 400

app = Flask(__name__)
app.config['SECRET_KEY'] = "63d3a56a02274adfa13333ec861fff32"
# all model configuration can go here
model_config = {
  "model_version": "v1",
  "tokenizer_version": "v1",
}


mp = ModelDependency(model_config)


def token_required(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        token = request.form.get('auth_key')
        if not token:
            return jsonify({"message": "토큰 필요"}), 403
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        except Exception as e:
            print(e)
            return jsonify({"message": "미인증 토큰"}), 403
        return func(*args, **kwargs)
    return decorator

# POST method
# request: auth key, jd text
# response: 직군
@app.route("/predict", methods=["POST"])
@token_required
def predict(): 
    try:
        jd = request.form.get('jd_text')
        jd = json.loads(jd)
        x = mp.prepare_text(jd)
        pred = mp.model.predict(x)
        print(pred)
        res = index_to_cat[argmax(pred)]
        return jsonify({"직군": res}), 200
    except JSONDecodeError:
        return jsonify({"message": "요청이 잘못 됐어요"}), 403
    except:
        return jsonify({"message": "서버 에러"}), 500



if __name__ == "__main__":
    app.run()
