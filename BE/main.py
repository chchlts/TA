from app import app
from flask import render_template, flash, redirect, url_for
from flask import Blueprint, Response, request, json
from flask_cors import CORS
import logging
from flask import request
import io
import sys
import numpy as np
import cv2
import os
from PIL import Image
import base64
import tensorflow as tf
import keras
from keras.utils import load_img, img_to_array

from keras import applications
from keras.applications.vgg16 import preprocess_input,decode_predictions,VGG16
from keras.models import load_model

from models.segmentation.models.unets import Unet2D
from models.segmentation.models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D

from models.segmentation.utils.learning.metrics import dice_coef, precision, recall
from models.segmentation.utils.io.data import normalize
from models.segmentation.utils.postprocessing.hole_filling import fill_holes
from models.segmentation.utils.postprocessing.remove_small_noise import remove_small_areas


cors = CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}})

# Setting
UPLOAD_FOLDER = 'static/uploads/'

app.secret_key ="ichaa-imut"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
weight_file_name = 'models/segmentation/training_history/2019-12-19 01_3A53_3A15.480800.hdf5'
treshold = 120


# Define model
model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
model = load_model(weight_file_name,
                 custom_objects={'recall':recall,
                                 'precision':precision,
                                 'dice_coef': dice_coef,
                                 'relu6':relu6,
                                 'DepthwiseConv2D':DepthwiseConv2D,
                                 'BilinearUpsampling':BilinearUpsampling})

filename = 'bbf82a8180020c2a8b38e6f8c7c86d7d_0.png'
# Helper function 
def render_frame(arr: np.ndarray):
    mem_bytes = io.BytesIO()
    img = Image.fromarray(arr)
    img.save(mem_bytes, 'JPEG')
    mem_bytes.seek(0)
    img_base64 = base64.b64encode(mem_bytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)

    return uri


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/segmented/' + filename), code=301)

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        flash('No file')
        return redirect(request.url)

    imagefile = request.files['imagefile']
    filename = imagefile.filename

    # Read the image file
    file_data = imagefile.read()

    # Convert the file data to a NumPy array
    nparr = np.frombuffer(file_data, np.uint8)

    # Decode the image using cv2.imdecode
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    padded_image = Image.new("RGB", [224, 224])
    padded_image.paste(Image.fromarray(image), (0, 0, image.shape[1], image.shape[0]))

    arr = np.array(padded_image)
    diff = np.amax(arr) - np.amin(arr)
    diff = 255 if diff == 0 else diff
    arr = arr / np.absolute(diff)
    arr = arr.reshape(1, input_dim_x, input_dim_y, 3)

    # Prediction
    prediction = model.predict(arr)
    pred = prediction[0] * 255
    segmented_image = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
    _ , threshed = cv2.threshold(segmented_image, 120, 255, type=cv2.THRESH_BINARY)
    filled = fill_holes(threshed, 120, 0.1) 
    denoised = remove_small_areas(filled, 120, 0.05)
    

    # Save segmented result -> overlay the masking prediction to actual image 
    img, bg_img = save_segmented_result(image,denoised)

    response = predict_score(img, bg_img)
    print(response, file=sys.stderr)

    return Response(
            response = json.dumps({
                "success": True,
                "message": "Success",
                "code": 200,
                "data": response
            }),
            status=200,
            mimetype="application/json"
        )

def padding(image):
   padded_image = Image.new("L", [224, 224])
   padded_image.paste(image, (0,0,image.width, image.height))

   return padded_image

def predict_score(img, bg_img):
   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   img = cv2.resize(img, (100, 100))
   img = img/255
   img = img.reshape(1, 100,100,3)
   
   bg_img = cv2.cvtColor(bg_img,cv2.COLOR_BGR2RGB)
   bg_img = cv2.resize(bg_img, (100, 100))
   bg_img = bg_img/255
   bg_img = bg_img.reshape(1, 100,100,3)

   color_model_weight = 'models\classification\color\color-model-best.h5'
   color_model = load_model(color_model_weight)

   pred = color_model.predict(img)
   print(pred, file=sys.stderr)
   
   threshold = 0.5
   predictions_bin = (pred[0] > threshold).astype(int)
   color_score = get_score(predictions_bin)
   color_desc = get_color_desc(int(color_score[0]))
   
   inflammation_model_weight = 'models\classification\inflammation\inflammation-model-best.h5'
   inflammation_model = load_model(inflammation_model_weight)

   inflammation_pred = inflammation_model.predict(bg_img)
   threshold = 0.35
   inflammation_predictions_bin = (inflammation_pred[0] > threshold).astype(int)
   inflammation_score = get_score(inflammation_predictions_bin)
   inflammation_desc = get_inflammation_desc(int(inflammation_score[0]))

   exudate_model_weigth = 'models\classification\exudate\model-best.h5'
   exudate_model = load_model(exudate_model_weigth)
   
   exudate_pred = exudate_model.predict(img)
   exudate_prediction_bin = (exudate_pred[0] > threshold).astype(int)
   exudate_score = get_score(exudate_prediction_bin)
   exudate_desc = get_exudate_score(int(exudate_score[0]))

   total_score = color_score[0] + inflammation_score[0] + exudate_score[0]
   print(total_score, file=sys.stderr)
   status = get_status(total_score)

   return {
       "predictions": {
            "color": color_score[0].tolist(),
            "color_desc" : color_desc,
            "inflammation": inflammation_score[0].tolist(),
            "inflammation_desc" : inflammation_desc,
            "exudate": exudate_score[0].tolist(),
            "exudate_desc": exudate_desc
        },
        "total": int(total_score),
        "status": status
   }

def get_color_desc(score):
   desc = "Undefined"
   if(score == 7):
      desc = "Warna HITAM (Nekrotik)"
   elif(score == 6):
      desc = "Warna HITAM-KUNING (Nekrotik disertai slough)"
   elif(score == 5):
      desc = "Warna KUNING (SLough)"
   elif(score == 4):
      desc = "Warna KUNING-MERAH ( Slough disertai granulasi )"
   elif( score == 3):
      desc = "Warna MERAH (Granulasi)"
   elif( score == 2):
      desc = "Warna MERAH-PINK (Granulasi disertai epitel)"
   elif(score == 1):
      desc = "Warna PINK (Epitel)"
   else :
      desc = "Undefined"
   
   return desc

def get_inflammation_desc(score):
   desc = "Undefined"
   if(score == 4):
      desc = "Inflamasi LUAS (Bengkak dan kemerahan yang nyata meluas dan merata di sekitar luka atau didapatkan pus  nanah dari luka)"
   elif( score == 3):
      desc = "Inflammasi SEDANG (Bengkak dan kemerahan tidak terlalu luas dan tidak merata di sektitar luka)"
   elif( score == 2):
      desc = "Inflammasi RINGAN (Ada kemerahan terbatas di tepi / di pinggir luka)"
   elif(score == 1):
      desc = "Inflamasi TIDAK ADA (Tidak tampak ada kemerahan di kulit sekitar luka)"
   else :
      desc = "Undefined"
   return desc

def get_exudate_score(score):
   desc = "Undefined"
   if(score == 5):
      desc = "Eksudat BANYAK (Basah membanjiri luka dan sekitarnya/balutan jenuh dan basah tidak dapat menampung dan tidak bertahan lama)"
   elif(score == 4):
      desc = "Eksudat SEDANG (Basah tampak pada bed luka / balutan basah masih dapat menampung)"
   elif( score == 3):
      desc = "Eksudat SEDIKIT (Tampak agak basah memantulkan cahaya/ balutan agak basah)"
   elif( score == 2):
      desc = "Eksudat KERING (Permukaan luka kering)"
   elif(score == 1):
      desc = "Eksudat LEMBAB (Tidak basah dan tidak kering)"
   else :
      desc = "Undefined"
   
   return desc

def get_score(pred):
   
   array = np.array(pred)

   # Get the indices where the array is 1
   indices = np.nonzero(array)[0]

   return indices+1

def get_status(score):
   if score >= 10 and score <=16 :
      status = 'Berat'
   elif score >=8 and score <=9 :
      status = 'Sedang'
   elif score >= 5 and score <= 7:
      status = 'Ringan'
   elif score >= 3 and score <= 4 :
      status = 'Sembuh'
   else :
      status = 'tidak terdefinisi' 
    
   return status

def extract_countour(image):
   image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   
   # red color boundaries [B, G, R]
   lower = np.array([10,10,10])
   upper = np.array([250, 250, 250])
   
   # find the colors within the specified boundaries and apply
   mask = cv2.inRange(image, lower, upper)
   output = cv2.bitwise_and(image, image, mask=mask)
   
   ret,thresh = cv2.threshold(mask, 40, 255, 0)
   contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),5)
   foreground = image[y:y+h,x:x+w]

   return foreground

def save_segmented_result(img,pred):
    pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    rows,cols,_ = img.shape
    bg_img = img.copy()
    for row in range(rows):
      for col in range(cols):
        k = pred[row,col]
        if(k != 255):
          img[row,col] = 0
        else:
           bg_img[row,col] = 0

    img = extract_countour(img)
    bg_img = extract_countour(bg_img)

    return img,bg_img
    
          
if __name__ == "__main__":
    app.run()