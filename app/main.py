import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from flask import Flask
from flask import jsonify, request
import numpy as np
import heapq
import cv2

app = Flask(__name__)




def topacc(imagepath):
    model = load_model(r'model')

    categories = ('alstroemeria', 'anemone', 'anthurium', 'arumlily', 'baloon flower', 'bellisdaisy', 'birdofparadise',
              'bouvardia', 'cherryblossom', 'coneflower',
              'cornflower', 'cypress', 'daffodil', 'dahlia', 'daisy', 'dandelion', 'dandelion',
              'edelweiss flower', 'foxglove', 'gazania',
              'hibicus', 'honeysuckle', 'hydrangea', 'iris', 'jasminum polyanthum', 'jasminum sambac', 'lantana',
              'laurel', 'lilac', 'lilies',
              'lilyofthevalley', 'lotus', 'loveinthemist', 'lupin', 'morningglory', 'myosotis', 'myrtus', 'orchid',
              'pansy', 'plumeria',
              'poinsettia', 'protea', 'ranunculus', 'rose', 'spearthistle', 'sunflower', 'tansy', 'tulip',
              'waterlilies', 'whiteclover', 'yarrow')
    x = np.expand_dims(imagepath, axis=0)
    imagepath = cv2.cvtColor(imagepath, cv2.COLOR_BGR2RGB)
    imagepath = cv2.resize(imagepath, (224, 224)).astype('float16')
    preds = model.predict(np.expand_dims(imagepath, axis=0))
    preds = np.array(preds).mean(axis=0)
    return preds


@app.route('/', methods=['GET', 'POST'])
def uploadfile():
    print("success")
    file = request.files['image'].read()
    print(file)
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    try:
        result = str(topacc(img))
        print(result)
        return (result)
    except:
        print('failed')
    return "DONE:DONE"




@app.route('/hello')
def hello_world():
    return "<h1>Welcome to Summoner's Drift</h1>"
