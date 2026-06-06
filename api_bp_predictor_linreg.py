import joblib

model = joblib.load("age_bp_predictor_lin_reg.pkl")

new_bp = model.predict([[160]])
print("Predicted bp = ", new_bp[0][0])