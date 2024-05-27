from flask import Flask, request, jsonify
import joblib
from modules.helpers import preprocess_image, get_output_category, decode_image

app = Flask(__name__)

@app.route('/image', methods=['POST', 'GET'])
def index():
    # get the encoded image in json format and decode it
    encoded_image = request.json.get('image')
    decoded_image = decode_image(encoded_image)
    
    # preprocess the image and input it to the model
    model_input = preprocess_image(decoded_image)
    model = joblib.load('vgg_model.joblib')
    model_output = model.predict(model_input)

    # return the final prediction
    final_prediction = get_output_category(model_output)

    return jsonify({'predicted category': final_prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
