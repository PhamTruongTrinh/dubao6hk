import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])
    int(output)
    if output == 1:
        return render_template('index.html', prediction_text='Chúc mừng rất có thể bạn sẽ ra trường đúng hạn')
    if output == 0:
        return render_template('index.html', prediction_text='Có thể bạn sẽ ra trường không đúng hạn. Hãy cố gắng hơn trong học tập')

if __name__ == "__main__":
    app.run(debug=True)
