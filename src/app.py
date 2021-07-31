from flask import Flask, flash, request, render_template, Response, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

################################ VGG ZONE ########################################
from matplotlib import pyplot
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
train_data = np.load(open('./src/data.stupid', 'rb'))
print('loaded model')

def extract_face(filename, required_size=(224, 224)):
	pixels = pyplot.imread(filename)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array


def get_embeddings(filenames):
	faces = [extract_face(f) for f in filenames]
	samples = np.asarray(faces, 'float32')
	samples = preprocess_input(samples, version=2)
	yhat = model.predict(samples)
	yhat = np.concatenate((train_data, yhat))
	return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	score = cosine(known_embedding, candidate_embedding)
	return score 

def test(new_filename):
	filenames = ["./src/images/" + new_filename]
	# filenames = ['./src/images/prawit.jpg', './src/images/prayut.jpg', "./src/images/anutin.jpg", "./src/images/" + new_filename]
	embeddings = get_embeddings(filenames)
	_results = {
        'prawit': (is_match(embeddings[0], embeddings[3])),
        'prayut': (is_match(embeddings[1], embeddings[3])),
        'anutin': (is_match(embeddings[2], embeddings[3]))
    }
	return _results

##################################################################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = './src/images'

app  = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, world!'

@app.route('/OmmNusctxunYpHqnCVnq', methods=['POST'])
@cross_origin()
def upload():
    print('new request!', request.files, request.files.to_dict())
    try:
        file = request.files.to_dict()['files']

        if file.filename == '':
            data = {
                'code': 'no such file'
            }
            response = app.response_class(
                response=json.dumps(data), 
                status=422, 
                mimetype='application/json'
            )

        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timeStamp = getTime()
            new_filename = secure_filename(newFileName(filename, 'picture'))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(filepath)
            # res = process_face_rec(new_filename)
            res = test(new_filename)
            data = {
                'code': 'success',
                'res': res,
                'timestamp': getTime()
            }
            response = app.response_class(
                response = json.dumps(data), 
                status   = 200, 
                mimetype = 'application/json'
            )
        else :
            data = {
                'code': 'file extension not allow'
            }
            response = app.response_class(
                response=json.dumps(data), 
                status=422, 
                mimetype='application/json'
            )
    except:
        data = {
            'code': 'bad request'
        }
        response = app.response_class(
            response=json.dumps(data), 
            status=400, 
            mimetype='application/json'
        )

    return response

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def getTime():
    return datetime.now(tz=None).strftime("%d-%b-%Y_%H:%M:%S_%f")

def newFileName(filename, newName):
    return newName + '.' + filename.rsplit('.', 1)[1]
