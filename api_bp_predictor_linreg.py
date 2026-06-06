import joblib
from flask import Flask,request

app = Flask(__name__)
model = joblib.load("age_bp_predictor_lin_reg.pkl")

@app.route("/test", methods = ["GET"])
def test():
    return "hello"

@app.route("/user", methods = ["POST"])
def user():
    input = request.get_json()
    print(input)
    age = input.get("age")
    print(age)
    new_bp = model.predict([[age]])
    print("Predicted bp = ", new_bp[0][0])
    return {"bp": new_bp[0][0]}

if __name__ == "__main__":
    app.run()
