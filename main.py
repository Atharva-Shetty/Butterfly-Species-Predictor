from flask import Flask , request, render_template

from tensorflow.keras.models import load_model
import os
import random

import tensorflow as tf

from werkzeug.utils import secure_filename

import os


app = Flask(__name__)

model = load_model('./butterflymodel.h5')
class_names = ['ADONIS',
 'AFRICAN GIANT SWALLOWTAIL',
 'AMERICAN SNOOT',
 'AN 88',
 'APPOLLO',
 'ARCIGERA FLOWER MOTH',
 'ATALA',
 'ATLAS MOTH',
 'BANDED ORANGE HELICONIAN',
 'BANDED PEACOCK',
 'BANDED TIGER MOTH',
 'BECKERS WHITE',
 'BIRD CHERRY ERMINE MOTH',
 'BLACK HAIRSTREAK',
 'BLUE MORPHO',
 'BLUE SPOTTED CROW',
 'BROOKES BIRDWING',
 'BROWN ARGUS',
 'BROWN SIPROETA',
 'CABBAGE WHITE',
 'CAIRNS BIRDWING',
 'CHALK HILL BLUE',
 'CHECQUERED SKIPPER',
 'CHESTNUT',
 'CINNABAR MOTH',
 'CLEARWING MOTH',
 'CLEOPATRA',
 'CLODIUS PARNASSIAN',
 'CLOUDED SULPHUR',
 'COMET MOTH',
 'COMMON BANDED AWL',
 'COMMON WOOD-NYMPH',
 'COPPER TAIL',
 'CRECENT',
 'CRIMSON PATCH',
 'DANAID EGGFLY',
 'EASTERN COMA',
 'EASTERN DAPPLE WHITE',
 'EASTERN PINE ELFIN',
 'ELBOWED PIERROT',
 'EMPEROR GUM MOTH',
 'GARDEN TIGER MOTH',
 'GIANT LEOPARD MOTH',
 'GLITTERING SAPPHIRE',
 'GOLD BANDED',
 'GREAT EGGFLY',
 'GREAT JAY',
 'GREEN CELLED CATTLEHEART',
 'GREEN HAIRSTREAK',
 'GREY HAIRSTREAK',
 'HERCULES MOTH',
 'HUMMING BIRD HAWK MOTH',
 'INDRA SWALLOW',
 'IO MOTH',
 'Iphiclus sister',
 'JULIA',
 'LARGE MARBLE',
 'LUNA MOTH',
 'MADAGASCAN SUNSET MOTH',
 'MALACHITE',
 'MANGROVE SKIPPER',
 'MESTRA',
 'METALMARK',
 'MILBERTS TORTOISESHELL',
 'MONARCH',
 'MOURNING CLOAK',
 'OLEANDER HAWK MOTH',
 'ORANGE OAKLEAF',
 'ORANGE TIP',
 'ORCHARD SWALLOW',
 'PAINTED LADY',
 'PAPER KITE',
 'PEACOCK',
 'PINE WHITE',
 'PIPEVINE SWALLOW',
 'POLYPHEMUS MOTH',
 'POPINJAY',
 'PURPLE HAIRSTREAK',
 'PURPLISH COPPER',
 'QUESTION MARK',
 'RED ADMIRAL',
 'RED CRACKER',
 'RED POSTMAN',
 'RED SPOTTED PURPLE',
 'ROSY MAPLE MOTH',
 'SCARCE SWALLOW',
 'SILVER SPOT SKIPPER',
 'SIXSPOT BURNET MOTH',
 'SLEEPY ORANGE',
 'SOOTYWING',
 'SOUTHERN DOGFACE',
 'STRAITED QUEEN',
 'TROPICAL LEAFWING',
 'TWO BARRED FLASHER',
 'ULYSES',
 'VICEROY',
 'WHITE LINED SPHINX MOTH',
 'WOOD SATYR',
 'YELLOW SWALLOW TAIL',
 'ZEBRA LONG WING']


def model_predict(image_path , model):
    filepath = image_path
    img = load_and_prep_image(filepath, scale = False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0)) 
    pred_class = class_names[pred_prob.argmax()] 
    return pred_class

def load_and_prep_image(filename, img_shape=224, scale=True):
  
  img = tf.io.read_file(filename)

  img = tf.io.decode_image(img)

  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:

    return img/255.
  else:
    return img


@app.route('/' , methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload():
    if request.method == 'POST':
      
        f = request.files['imagefile']

      
        basepath = './images/'
        file_path = os.path.join(
            basepath,  secure_filename(f.filename))
        
        f.save(file_path)
        

    
        pred = model_predict(file_path , model)
      
        res = pred.replace(' ' , '')
        path = './static/images/' + res
        
        
        
        files=os.listdir(path)
        d=random.choice(files)
        res = res.replace(' ' , '')
   
        random_img = '../static/images/' + res + '/' + d   
        
       
        
        
        return render_template('index.html' , prediction=pred , image=random_img )
        

if __name__ == '__main__':
    app.run(debug=True)
