from flask import Flask,request,jsonify
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)
X,y = load_iris(return_X_y = True)
model = LogisticRegression(max_iter=200).fit(X,y)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction':int(prediction[0])})
if __name__ == '__main__':
    app.run(debug=True)
