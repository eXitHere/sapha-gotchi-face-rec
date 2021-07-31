from matplotlib import pyplot
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

model = VGGFace(model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
train_data = np.load(open('./src/data.stupid', 'rb'))
print('loaded model!')

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
	print('Eiei')
	yhat = model.predict(samples)
	print('Yes')
	yhat = np.concatenate((train_data, yhat))
	return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.5):
	score = cosine(known_embedding, candidate_embedding)
	return score 

def test(new_filename):
	filenames = ["./src/images/" + new_filename]
	# filenames = ['./src/images/prawit.jpg', './src/images/prayut.jpg', "./src/images/anutin.jpg", "./src/images/" + new_filename]
	embeddings = get_embeddings(filenames)
	print(embeddings)
	_results = {
        'prawit': (is_match(embeddings[0], embeddings[3])),
        'prayut': (is_match(embeddings[1], embeddings[3])),
        'anutin': (is_match(embeddings[2], embeddings[3]))
    }
	return _results