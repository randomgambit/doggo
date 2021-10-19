

from typing import Dict

import streamlit as st
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from IPython.display import SVG
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

import tensorflow

from PIL import Image

mylist = {0: 'Chihuahua',
 1: 'Japanese_spaniel',
 2: 'Maltese_dog',
 3: 'Pekinese',
 4: 'Tzu',
 5: 'Blenheim_spaniel',
 6: 'papillon',
 7: 'toy_terrier',
 8: 'Rhodesian_ridgeback',
 9: 'Afghan_hound',
 10: 'basset',
 11: 'beagle',
 12: 'bloodhound',
 13: 'bluetick',
 14: 'tan_coonhound',
 15: 'Walker_hound',
 16: 'English_foxhound',
 17: 'redbone',
 18: 'borzoi',
 19: 'Irish_wolfhound',
 20: 'Italian_greyhound',
 21: 'whippet',
 22: 'Ibizan_hound',
 23: 'Norwegian_elkhound',
 24: 'otterhound',
 25: 'Saluki',
 26: 'Scottish_deerhound',
 27: 'Weimaraner',
 28: 'Staffordshire_bullterrier',
 29: 'American_Staffordshire_terrier',
 30: 'Bedlington_terrier',
 31: 'Border_terrier',
 32: 'Kerry_blue_terrier',
 33: 'Irish_terrier',
 34: 'Norfolk_terrier',
 35: 'Norwich_terrier',
 36: 'Yorkshire_terrier',
 37: 'haired_fox_terrier',
 38: 'Lakeland_terrier',
 39: 'Sealyham_terrier',
 40: 'Airedale',
 41: 'cairn',
 42: 'Australian_terrier',
 43: 'Dandie_Dinmont',
 44: 'Boston_bull',
 45: 'miniature_schnauzer',
 46: 'giant_schnauzer',
 47: 'standard_schnauzer',
 48: 'Scotch_terrier',
 49: 'Tibetan_terrier',
 50: 'silky_terrier',
 51: 'coated_wheaten_terrier',
 52: 'West_Highland_white_terrier',
 53: 'Lhasa',
 54: 'coated_retriever',
 55: 'coated_retriever',
 56: 'golden_retriever',
 57: 'Labrador_retriever',
 58: 'Chesapeake_Bay_retriever',
 59: 'haired_pointer',
 60: 'vizsla',
 61: 'English_setter',
 62: 'Irish_setter',
 63: 'Gordon_setter',
 64: 'Brittany_spaniel',
 65: 'clumber',
 66: 'English_springer',
 67: 'Welsh_springer_spaniel',
 68: 'cocker_spaniel',
 69: 'Sussex_spaniel',
 70: 'Irish_water_spaniel',
 71: 'kuvasz',
 72: 'schipperke',
 73: 'groenendael',
 74: 'malinois',
 75: 'briard',
 76: 'kelpie',
 77: 'komondor',
 78: 'Old_English_sheepdog',
 79: 'Shetland_sheepdog',
 80: 'collie',
 81: 'Border_collie',
 82: 'Bouvier_des_Flandres',
 83: 'Rottweiler',
 84: 'German_shepherd',
 85: 'Doberman',
 86: 'miniature_pinscher',
 87: 'Greater_Swiss_Mountain_dog',
 88: 'Bernese_mountain_dog',
 89: 'Appenzeller',
 90: 'EntleBucher',
 91: 'boxer',
 92: 'bull_mastiff',
 93: 'Tibetan_mastiff',
 94: 'French_bulldog',
 95: 'Great_Dane',
 96: 'Saint_Bernard',
 97: 'Eskimo_dog',
 98: 'malamute',
 99: 'Siberian_husky',
 100: 'affenpinscher',
 101: 'basenji',
 102: 'pug',
 103: 'Leonberg',
 104: 'Newfoundland',
 105: 'Great_Pyrenees',
 106: 'Samoyed',
 107: 'Pomeranian',
 108: 'chow',
 109: 'keeshond',
 110: 'Brabancon_griffon',
 111: 'Pembroke',
 112: 'Cardigan',
 113: 'toy_poodle',
 114: 'miniature_poodle',
 115: 'standard_poodle',
 116: 'Mexican_hairless',
 117: 'dingo',
 118: 'dhole',
 119: 'African_hunting_dog'}

train_data_dir = r"."

@st.cache(show_spinner=False)
def mypred(myloaded):
    model = keras.models.load_model(r"mymodel.h5")

    img_width, img_height = 224, 224 
    channels = 3
    batch_size = 64
    num_images= 50
    image_arr_size= img_width * img_height * channels
    
        
    train_datagen = ImageDataGenerator(
        rescale= 1./255,
        shear_range= 0.2,
        zoom_range= 0.2,
        horizontal_flip= True,
        rotation_range= 20,
        width_shift_range= 0.2,
        height_shift_range= 0.2,   
        validation_split=0.2,
    
    )
    
    train_generator = train_datagen.flow_from_directory(  
        train_data_dir,  
        target_size= (img_width, img_height), 
        color_mode= 'rgb',
        batch_size= batch_size,  
        class_mode= 'categorical',
        subset='training',
        shuffle= True, 
        seed= 1337
    ) 

    data = img_to_array(myloaded)
    samples = expand_dims(data, 0)
    plt.imshow(myloaded)
    it = train_datagen.flow(samples, batch_size=1)
    y_prob =model.predict(it)
    len(y_prob[0])
    y_classes = y_prob[0].argmax(axis=-1)
    return mylist[y_classes]


if __name__ == "__main__":
    
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")


    with col3:
        st.write("")

    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
    
    if uploaded_file is not None:
        myimg = Image.open(uploaded_file)
        myimg.save('myimage.png')
        outcome = mypred(myimg)
        with col2:
            myformatname = f"""Your dog is likely a <p style="font-family:sans-serif; color:Green; font-size: 24px;">{outcome.replace('_', ' ')}</p>"""
            st.markdown(myformatname, unsafe_allow_html=True)
            st.image(myimg, caption = f"Your doggo")


