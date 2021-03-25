flask_debug=1
from flask import Flask, render_template, request, redirect, url_for,send_file,send_from_directory
import numpy as np
#from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
#from tqdm import tqdm 
import zipfile
import pathlib
from os import listdir
from matplotlib import image
import os
import tensorflow as tf
from keras import Input, Model, Sequential
#from keras.layers import Conv2D, MaxPooling2D, Concatenate, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D,Dense,concatenate,Activation,Dropout,Input
from tqdm import tqdm
from werkzeug.utils import secure_filename
import shutil
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



UPLOAD_FOLDER1 = 'static/uploads/'
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
###################################################

###################################################
#https://buildcoding.com/upload-and-download-file-using-flask-in-python/
#https://roytuts.com/how-to-download-file-using-python-flask/
#timestr = time.strftime("%Y%m%d-%H%M%S")
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/download')
def download():
   return render_template('download.html')

@app.route('/view')
def view():
   return render_template('view.html')
   


@app.route('/downloads')
def download_file():
   path = "download\\People_Count_Prediction.csv"
   return send_file(path, as_attachment=True,cache_timeout=0)
   #return flask.render_template('download.html')

@app.route('/Error')
def Error():
   return render_template('Error.html')
  
@app.route('/Errors')
def Errors():
   path = "Errors.txt"
   return send_file(path, as_attachment=True,cache_timeout=0)
   #return flask.render_template('download.html')



@app.route('/predict', methods=['POST'])
def predict():
    os.mkdir("./uploads") 

    print ("inside predict"+request.method)
    if request.method == 'POST':
        import tensorflow as tf
        AUTOTUNE = tf.data.AUTOTUNE

        print("requestmethod")
        
        
        uploaded_file = request.files['file']
        #uploaded_file_name = request.files['file'].name
        print(uploaded_file.filename)
        print ("inside predict1234")
        if not uploaded_file.filename.endswith('.zip'):
            f = open("Errors.txt", "w")
            f.write("Uploaded file is not zip file. Please upload zip file and try again")
            f.close()
            return  redirect(url_for('Error'))
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #uploaded_file.save(uploaded_file.filename)
        #extraact zip file 
        with zipfile.ZipFile("./uploads/"+filename, 'r') as zip_ref:
            zip_ref.extractall("./uploads")    
        data_dir="./uploads"
        
        
        
       
        # load all images in a directory
        loaded_images = list()
        filenames=[]
        for filename in listdir('uploads'):
            # load image
            print(filename)
            if filename.endswith('.jpg'):
                #print(filename, img_data.shape)
                img_data = image.imread('uploads/' + filename)
                # store loaded image
                loaded_images.append(img_data)
                filenames.append(filename)
            elif (not filename.endswith('.jpg') and not filename.endswith('.zip')):
                f = open("Errors.txt", "w")
                f.write("Zip file contains non .jpg files. Please check and try again with .jpg files ")
                f.close()
                return  redirect(url_for('Error'))
        np.save(os.path.join('./uploads','uploads'),np.array(loaded_images))
        test_data = np.load("./uploads/uploads.npy")
        test_data = tf.data.Dataset.from_tensor_slices((test_data))
        seed=(1,2)
        def preprocess_image(image):
            #image,count = image,count
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [400, 400])  #resizing the image to 400X400 pixels
            image = tf.image.random_flip_left_right(image) #fliping the image left to right
            image = tf.image.stateless_random_flip_up_down(image,seed) #flipping the image upside down

            image = tf.image.stateless_random_brightness(image, max_delta=32.0 / 255.0, seed=seed) #Randomly changing the brightness to the images
            image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed) #Randomly changing saturation to the images

            return image
        def configure_for_performance(ds):
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds
            
        batch_size=1
        input_shape=(400, 400,3)
        test_data = test_data.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        test_data = configure_for_performance(test_data)
        
        
        # Importing tensorflow
        np.random.seed(42)
        import tensorflow as tf
        #tf.set_random_seed(42)
        tf.random.set_seed(42)

        # Configuring a session
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3
        )

        # Import Keras
        from keras import backend as K
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        #K.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)


        BestModel= tf.keras.models.load_model('Model2.h5')
        y_test_results = BestModel.predict(test_data)

        y_test_results_final=[]
        for res in y_test_results:
            y_test_results_final.append(res[0])
           

        #Post Quantization
        interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()[0]["index"]
        output_details = interpreter.get_output_details()[0]["index"]
        interpreter.get_input_details()



        y_test_quantization_results_model2=[]
        # Run predictions on every image in the "test" dataset.

        for img in tqdm(test_data):
          interpreter.set_tensor(input_details, img)
          # Run inference.
          interpreter.invoke()
          # Post-processing
          predictions  = interpreter.get_tensor(output_details)
          y_test_quantization_results_model2.append(predictions)
          
        y_test_quantization_results_model2_final = []
        for ikm in range(len(test_data)):
            y_test_quantization_results_model2_final.append(y_test_quantization_results_model2[ikm][0][0])




        
        test_results = pd.DataFrame()
        test_results['FileName'] = filenames
        test_results['Predicted_Count']=np.round(y_test_results_final)
        test_results['Predicted(Post Quantization)_Count']=np.round(y_test_quantization_results_model2_final)
                 

        test_results.to_csv("download\People_Count_Prediction.csv",index=False)   
        
        #os.rmdir("./uploads")
        shutil.rmtree('./uploads')

        return redirect(url_for('download'))
    return render_template('download.html')


@app.route('/predictoneimage', methods=['POST'])
def predictoneimage():
    os.mkdir("./uploads") 

    print ("inside predictoneimage"+request.method)
    if request.method == 'POST':
        import tensorflow as tf
        AUTOTUNE = tf.data.AUTOTUNE

        print("requestmethod")
        
        
        uploaded_file = request.files['file']
        #uploaded_file_name = request.files['file'].name
        print(uploaded_file.filename)
        print ("inside predict1234")
        if not uploaded_file.filename.endswith('.jpg'):
            f = open("Errors.txt", "w")
            f.write("Uploaded file is not zip file. Please upload zip file and try again")
            f.close()
            return  redirect(url_for('Error'))
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #uploaded_file.save(uploaded_file.filename)

        
        
        
       
        # load all images in a directory
        loaded_images = list()
        filenames=[]
        for filename in listdir('uploads'):
            # load image
            print(filename)
            if filename.endswith('.jpg'):
                #print(filename, img_data.shape)
                img_data = image.imread('uploads/' + filename)
                # store loaded image
                loaded_images.append(img_data)
                filenames.append(filename)
            elif (not filename.endswith('.jpg') and not filename.endswith('.zip')):
                f = open("Errors.txt", "w")
                f.write("Zip file contains non .jpg files. Please check and try again with .jpg files ")
                f.close()
                return  redirect(url_for('Error'))
        np.save(os.path.join('./uploads','uploads'),np.array(loaded_images))
        np.save(os.path.join('./static','uploads'),np.array(loaded_images))
        test_data = np.load("./uploads/uploads.npy")
        test_data = tf.data.Dataset.from_tensor_slices((test_data))
        seed=(1,2)
        def preprocess_image(image):
            #image,count = image,count
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [400, 400])  #resizing the image to 400X400 pixels
            image = tf.image.random_flip_left_right(image) #fliping the image left to right
            image = tf.image.stateless_random_flip_up_down(image,seed) #flipping the image upside down

            image = tf.image.stateless_random_brightness(image, max_delta=32.0 / 255.0, seed=seed) #Randomly changing the brightness to the images
            image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed) #Randomly changing saturation to the images

            return image
        def configure_for_performance(ds):
            ds = ds.cache()
            ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds
            
        batch_size=1
        input_shape=(400, 400,3)
        test_data = test_data.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        test_data = configure_for_performance(test_data)
        
        
        # Importing tensorflow
        np.random.seed(42)
        import tensorflow as tf
        #tf.set_random_seed(42)
        tf.random.set_seed(42)

        # Configuring a session
        session_conf = tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=3,
            inter_op_parallelism_threads=3
        )

        # Import Keras
        from keras import backend as K
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        #K.set_session(sess)
        tf.compat.v1.keras.backend.set_session(sess)


        BestModel= tf.keras.models.load_model('Model2.h5')
        y_test_results = BestModel.predict(test_data)

        y_test_results_final=[]
        for res in y_test_results:
            y_test_results_final.append(res[0])
           

        #Post Quantization
        interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()[0]["index"]
        output_details = interpreter.get_output_details()[0]["index"]
        interpreter.get_input_details()



        y_test_quantization_results_model2=[]
        # Run predictions on every image in the "test" dataset.

        for img in tqdm(test_data):
          interpreter.set_tensor(input_details, img)
          # Run inference.
          interpreter.invoke()
          # Post-processing
          predictions  = interpreter.get_tensor(output_details)
          y_test_quantization_results_model2.append(predictions)
          
        y_test_quantization_results_model2_final = []
        for ikm in range(len(test_data)):
            y_test_quantization_results_model2_final.append(y_test_quantization_results_model2[ikm][0][0])




        
        #test_results = pd.DataFrame()
        #test_results['FileName'] = filenames
        #test_results['Predicted_Count']=np.round(y_test_results_final)
        #test_results['Predicted(Post Quantization)_Count']=np.round(y_test_quantization_results_model2_final)
                 

        #test_results.to_csv("download\People_Count_Prediction.csv",index=False)   
        
        #os.rmdir("./uploads")
        shutil.rmtree('./uploads')

        #return redirect(url_for('view'))
    #return render_template('view.html')
        return render_template('view.html', filename=filenames[0], prediction_count=np.round(y_test_results_final[0]),prediction_countquantization=np.round(y_test_quantization_results_model2_final[0]))

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
#def downloads():
#    path = "simple.docx"#
	#path = "sample.txt"
	#return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)


