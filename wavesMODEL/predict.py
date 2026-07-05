import joblib

model = joblib.load("model.pkl")

def predict(features):
    result = model.predict([features])[0]
    return "ON" if result == 1 else "OFF"

if __name__ == "__main__":
    sample = [0.1,0.2,0.3,0.4,0.5,1.1,0.9,1.2,0.7]
    print("Prediction:", predict(sample))
