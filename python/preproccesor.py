import ast
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np  # Import NumPy
from itertools import combinations
import csv
import pandas as pd

from statistics import mean 

from scipy.stats import skew, kurtosis

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Activation, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K 
from datetime import datetime 

from keras import models
from keras import layers

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# ANGER = 3
# HAPPY = 2
# CALM = 1
# SAD = 0


# FOLDER 00 = SAD
# FOLDER 01 = 

features_names = ['tonnetz', 'chroma', 'rms', 'spec_flux', 'spec_cont', 'spec_cent', 'spec_band', 'roll_off', 'zcr']
stats = ['mean', 'var', 'std', 'median', 'min', 'max']
feature_data = {}
feature_columns = {'tonnetz': [1,6], 'chroma': [7,12], 'rms':[13,18], 'spec_flux':[19,24], 'spec_cont':[25,30], 'spec_cent':[31,36], 'spec_band':[37,42], 'roll_off':[43,48], 'zcr':[49,54]}#, 'mfcc':[55,174]}
max_dict = {}
avg_dict = {}


def get_feature_data(name, val, feature_data):
    '''
        Function: updates dictonary with a feature and its stats

        param name: Name of feature (tonnetz, spec_flux, etc..)
        param val: The value of the feature extracted from the song
        param feature_data: dictonary that is to be updated

        Returns: None
    '''
    feature_data[name,'mean'] = np.mean(val)
    feature_data[name,'var'] = np.var(val)
    feature_data[name,'std'] = np.std(val)
    # feature_data[name,'skew'] = skew(val)
    # feature_data[name,'kurtosis'] = kurtosis(val)
    feature_data[name,'median'] = np.median(val)
    feature_data[name,'min'] = np.min(val)
    feature_data[name,'max'] = np.max(val)




# Caculate MFCCS 
def mfcss_feature_extractor(audio_path):
    debussy, sr = librosa.load(audio_path)
    mfccs_debussy = librosa.feature.mfcc(y=debussy, sr=sr, n_mfcc=20)
    # mfcss_scaled_features = np.mean(mfccs_debussy.T,axis=0)
    return mfccs_debussy

# Calculate amplitude envelope with given frame size and hop length
def amplitude_envelope(signal, frame_size, hop_length):
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return amplitude_envelope 


### WRITE TO CSV ###
## Create a single csv file(csv_file) that contains features(features) from a folder of songs (artist_folder) from directory (directory) ###
def write_to_csv(features, csv_file, artist_folder, directory):
    header = 'filename tempo'

    # Add specific features to csv
    for f in features:
        for s in stats:
            feature_stat = f + '_' + s
            header += f' {feature_stat}'
    
    # Mfcc feature will be apart of any and all feature combos
    for i in range(1, 21):
        for s in stats:
            mfcc_stat = f'mfcc{i}' + '_' + s
            header += f' {mfcc_stat}'
    
    # header += ' label'
    header = header.split()

    file = open(csv_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
   
    # directory = 'songs_wav/'

    for sub_folder, folder, audio_files in os.walk(directory):
        if sub_folder == 'songs_wav/deam' or sub_folder == 'songs_wav/0505' or sub_folder == 'songs_wav/':
            continue
 
 
        if sub_folder == f'{directory}{artist_folder}':
            for audio in audio_files:
        
                    
                FRAME_SIZE = 1024
                HOP_LENGTH = 512
                print(f'Sub folder is: {sub_folder}\n Audio path is: {audio}')
                audio_path = sub_folder + '/' + audio
                debussy, sr = librosa.load(audio_path, res_type='kaiser_fast')

                
                # tempo
                tempo = librosa.feature.tempo(y=debussy,sr=sr)[0]

                # if 'tonnetz' in features:
                # tonnetz
                tonnetz = librosa.feature.tonnetz(y=debussy, sr=sr)
                get_feature_data('tonnetz', tonnetz)

                # if 'chroma' in features:
                # Chroma features
                chroma_stft = librosa.feature.chroma_stft(y=debussy, sr=sr)
                get_feature_data('chroma', chroma_stft)

                # if 'rms' in features:
                # Root mean square energy
                rms = librosa.feature.rms(y=debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                get_feature_data('rms', rms)

                # if 'spec_flux' in features:
                # spectral flux
                spec_flux = librosa.onset.onset_strength(y=debussy, sr=sr)
                get_feature_data('spec_flux', spec_flux)

                # if 'spec_cont' in features:
                # spectral contrast
                S = np.abs(librosa.stft(debussy))
                spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr)
                get_feature_data('spec_cont', spec_cont)

                # if 'spec_cent' in features:
                # spectral centroid 
                spec_cent = librosa.feature.spectral_centroid(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                get_feature_data('spec_cent', spec_cent)

                # if 'spec_band' in features:
                # spectral bandwith
                spec_band = librosa.feature.spectral_bandwidth(y=debussy, sr=sr)
                get_feature_data('spec_band', spec_band)

                

                # if 'roll_off' in features:
                # spectral roll off
                roll_off = librosa.feature.spectral_rolloff(y=debussy, sr=sr)[0]
                get_feature_data('roll_off', roll_off)
                
                # if 'zcr' in features:
                # zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y=debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                get_feature_data('zcr', zcr)
                
                
   
                # # Set label according to valence-arousal emotion model
                # label = sub_folder.split('/', 1)
                # match label[1]:
                #     case "00":
                #         label = "Sad"
                #     case "01":
                #         label = "Anger"
                #     case "10":
                #         label = "Calm"
                #     case "11":
                #         label = "Happy"
                #     case _:
                #         label = "No label"
        
                # Add mean, var, etc of each feature to csv
                to_append = f'{audio} {tempo}'
                for f in features:
                    for s in stats:
                        to_append += f' {feature_data[f,s]}'
                
                
                # if 'mfcc' in features:
                # mfcc
                mfcc = mfcss_feature_extractor(audio_path)
                for i in range(len(mfcc)):
                
                    to_append += f' {np.mean(mfcc[i])}'
                    to_append += f' {np.var(mfcc[i])}'
                    to_append += f' {np.std(mfcc[i])}'
                    to_append += f' {np.median(mfcc[i])}'
                    to_append += f' {np.min(mfcc[i])}'
                    to_append += f' {np.max(mfcc[i])}'


                    

                # to_append += f' {label}'
                file = open(csv_file, 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

        




def get_conv_model(x_train, x_test, y_train,y_test,num_layers,f1, f2, f3, k1, k2, k3, d, loss, metrics = 'accuracy'):
    # Number of emotions
    num_labels = y_train.shape[1]


    ## Build network
    K.clear_session()
    model = Sequential()




    model.add(Input(shape=(x_train.shape[1], 1)))

    for i in range(1,num_layers):
        model.add(Conv1D(f1, k1, activation='relu', padding='same'))
        model.add(BatchNormalization(name = f'BN{i}'))
        model.add(MaxPooling1D(pool_size=2,name = f'MaxPooling{i}'))
    
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    

   
    model.compile(loss=loss,metrics=[metrics],optimizer='adam')

    ## Train model
    earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=30,
                                        restore_best_weights=True,
                                        verbose=1)
    history = model.fit(x_train,
                        y_train,
                        epochs=150,
                        batch_size=8,
                        validation_data=(x_test, y_test),
                        callbacks=earlystopping)
    

    # y_pred = model.predict(x_test)
    # pred = np.argmax(y_pred, axis = 1)

  
    test_loss, test_acc = model.evaluate(x_test,y_test)
    return test_acc
    # return model
    



def feature_combo_test(x_train_scaled, x_test_scaled, y_train, y_test, num_combo):

    key_list = list(feature_columns.keys())
    val_list = list(feature_columns.values())
    

    #for i in range(1, len(feature_columns) + 1):  
    # Find all combinations of features with size 'num_combo'
    # num_combo = 6 -> Will return all combinations of size 6
    layer1_list = list(combinations(feature_columns.values(), num_combo))

 
 

  
    # x_train_combo = x_train_scaled[:,55:]
    # x_test_combo = x_test_scaled[:,55:]

    for j in range(0, len(layer1_list)):
        x_train_combo = x_train_scaled[:,55:]
        x_test_combo = x_test_scaled[:,55:]

        # List of the features that is being tested per each model run
        feature_combo = []
        
        # Combine features into training and testing dataset
        layer2_list = layer1_list[j]
        for k in range(0, len(layer2_list)):

            m,n = layer2_list[k]
            x_train_combo = np.concatenate( [x_train_combo, x_train_scaled[:,m:n]],axis =1  )
            x_test_combo = np.concatenate( [x_test_combo, x_test_scaled[:,m:n]],axis =1  )
            
            # Keep track which feature combo is currently being ran in model
            position = val_list.index([m,n])
            feature_combo.append(key_list[position])

        # Run model using feature combo
        tests = []
        for x in range(5):
            tests.append(get_conv_model(x_train_combo,x_test_combo,y_train,y_test,1,64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy'))
        
        # Get max accuracy from each model run
        max_acc = max(tests)
        if max_acc not in max_dict:
            max_dict[max_acc] = []
        max_dict[max_acc].append(feature_combo)

        # Get average accruacy from each model run
        avg_acc = mean(tests)
        if avg_acc not in avg_dict:
            avg_dict[avg_acc] = []
        avg_dict[avg_acc].append(feature_combo)


        # Write dictonary to csv
        with open(f'feature_combos/{num_combo}_feat_max.csv', 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'Feature Combo']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key in max_dict:
                writer.writerow({'Accuracy': key, 'Feature Combo': max_dict[key]})


        with open(f'feature_combos/{num_combo}_feat_avg.csv', 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'Feature Combo']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key in avg_dict:
                writer.writerow({'Accuracy': key, 'Feature Combo': avg_dict[key]})



    max_dict.clear()
    avg_dict.clear()
    # break

    
def layer_test(x_train_scaled, x_test_scaled, y_train, y_test, csv_file):
    key_list = list(feature_columns.keys())
    val_list = list(feature_columns.values())

    

    for num_file in range(7,10):
        max_csv = []
        avg_csv = []
        with open(f'feature_combos/{num_file}_feat_{csv_file}.csv') as file_obj: 
            # skips heading
            heading = next(file_obj)
        
            reader_obj = csv.reader(file_obj) 
    
            # Iterate over each row in the csv  
            # file using reader object 
            for row in reader_obj: 

                
                # Check if feature combination has accuracy of at least 80%
                if float(row[0]) > 0.79:

                    # initialize training and test sets to have mfcc 1-20
                    x_train_combo = x_train_scaled[:,55:]
                    x_test_combo = x_test_scaled[:,55:]
                    
                    # Convert csv row to list
                    L = row[1]
                    layer1 = ast.literal_eval(L)
                    
                    # loop through lists of feature combos
                    for i in range(len(layer1)):

                        
                        # List of the features that is being tested per each model run
                        feature_combo = []

                        layer2 = layer1[i]
                        
                        # loop through feature combo
                        for j in range(len(layer2)):
                            m,n = feature_columns[layer2[j]]

                            x_train_combo = np.concatenate( [x_train_combo, x_train_scaled[:,m:n]],axis =1  )
                            x_test_combo = np.concatenate( [x_test_combo, x_test_scaled[:,m:n]],axis =1  )

                            # Keep track which feature combo is currently being ran in model
                            position = val_list.index([m,n])
                            feature_combo.append(key_list[position])
                        
                        # Number of layers in each model run
                        for num_layer in range(1,5):

                            # For each model run, run 3 tests
                            tests = []
                            for x in range(3):
                                tests.append((num_layer,get_conv_model(x_train_combo,x_test_combo,y_train,y_test,num_layer,64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy')))  

                            # Get max accuracy from each model run
                            max_acc = max(tests, key=lambda x:x[1])
                            max_percent = max_acc[1]


                            # Get average accuracy from each model run
                            avg_acc = sum(map(lambda x: x[1], tests)) / len(tests)

                            max_csv.append([max_percent, num_layer, feature_combo])
                            avg_csv.append([avg_acc, num_layer, feature_combo])

        with open(f'layer_tests/{num_file}_feat_max.csv', 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'Layers', 'Feature Combo']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in max_csv:
                # if len(i[2]) == i:
                writer.writerow({'Accuracy': i[0],'Layers': i[1], 'Feature Combo': i[2]})

        with open(f'layer_tests/{num_file}_feat_avg.csv', 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'Layers', 'Feature Combo']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in avg_csv:
                writer.writerow({'Accuracy': i[0],'Layers': i[1], 'Feature Combo': i[2]})

 

                    
            


    # with open(f'layer_tests/max.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['Accuracy', 'Layers', 'Feature Combo']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     for i in max_csv:
    #         writer.writerow({'Accuracy': i[0],'Layers': i[1], 'Feature Combo': i[2]})

    # with open(f'layer_tests/avg.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['Accuracy', 'Layers', 'Feature Combo']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
    #     for i in avg_csv:
    #         writer.writerow({'Accuracy': i[0],'Layers': i[1], 'Feature Combo': i[2]})
                


def final_test(x_train_scaled, x_test_scaled, y_train, y_test, csv_file):
    avg_csv = []
    for num_file in range(1,10):
        
        with open(f'layer_tests/{num_file}_feat_{csv_file}.csv') as file_obj: 
            # skips heading
            heading = next(file_obj)
        
            reader_obj = csv.reader(file_obj) 
    
            # Iterate over each row in the csv  
            # file using reader object 
            for row in reader_obj: 

                
                # Check if feature combination has accuracy of at least 80%
                if float(row[0]) > 0.90:

                    # initialize training and test sets to have mfcc 1-20
                    x_train_combo = x_train_scaled[:,55:]
                    x_test_combo = x_test_scaled[:,55:]
                    
                    # Convert csv row to list
                    L = row[2]
                    feature_combo = ast.literal_eval(L)

                    layers = int(row[1])
                        
                    # loop through feature combo
                    for j in range(len(feature_combo)):
                        m,n = feature_columns[feature_combo[j]]

                        x_train_combo = np.concatenate( [x_train_combo, x_train_scaled[:,m:n]],axis =1  )
                        x_test_combo = np.concatenate( [x_test_combo, x_test_scaled[:,m:n]],axis =1  )

                    # For each model run, run 5 tests
                    tests = []
                    for x in range(6):
                        tests.append((layers,get_conv_model(x_train_combo,x_test_combo,y_train,y_test,layers,64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy')))  

  

                    # Get average accuracy from each model run
                    avg_acc = sum(map(lambda x: x[1], tests)) / len(tests)
                    avg_csv.append([avg_acc, layers, feature_combo])

  
                    
        

    with open('avg.csv', 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'Layers', 'Feature Combo']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
 
        writer.writeheader()
        for i in avg_csv:

            # if len(i[2]) == i:
            writer.writerow({'Accuracy': i[0],'Layers': i[1], 'Feature Combo': i[2]})



def predict( csv):
    '''
        Function: Uses saved model to predict the mood of songs

        param csv: a csv file of songs and their respective features
        param feature_list: a list of features that are to be extracted from each song

        Returns: the mood prediction of each song from the folder
    '''
    model = models.load_model("music_model.keras")



    data = pd.read_csv(csv)
    print(np.array(data.iloc[:, 0]))
    data = data.drop(['filename'],axis=1)
    data = data.drop(['tempo'],axis=1)

    x = np.array(data, dtype = float)
    

  
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    x_predict = model.predict(x_scaled) 
    pred = np.argmax(x_predict, axis = 1)
    return pred

  

def write_csv(features, folder):
    '''
        Function: Extracts spefific features from a song and adds it to a csv file

        param features: a list of features that are to be extracted from each song
        param folder: the folder which holds the songs to be extracted from

        Returns: A csv file of songs and their extracted features
    '''
    feature_data = {}
    csv_file = f'{folder}_features.csv'
    header = 'filename tempo'

    # Add specific features to csv
    for f in features:
        for s in stats:
            feature_stat = f + '_' + s
            header += f' {feature_stat}'
    
    # Mfcc feature will be apart of any and all feature combos
    for i in range(1, 21):
        for s in stats:
            mfcc_stat = f'mfcc{i}' + '_' + s
            header += f' {mfcc_stat}'
    
    # header += ' label'
    header = header.split()

    file = open(csv_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
   
    
    for audio in os.listdir(folder):
        
        if audio.endswith(".mp3") or audio.endswith(".wav"):
            audio_path = os.path.join(folder, audio)



            print(f'EXTRACTING {audio_path}')

                
            FRAME_SIZE = 1024
            HOP_LENGTH = 512

    
            debussy, sr = librosa.load(audio_path, res_type='kaiser_fast')

    

            
            # tempo
            tempo = librosa.feature.tempo(y=debussy,sr=sr)[0]


            # if 'tonnetz' in features:
            # tonnetz
            tonnetz = librosa.feature.tonnetz(y=debussy, sr=sr)
            get_feature_data('tonnetz', tonnetz, feature_data)


            # if 'chroma' in features:
            # Chroma features
            chroma_stft = librosa.feature.chroma_stft(y=debussy, sr=sr)
            get_feature_data('chroma', chroma_stft, feature_data)

            # if 'rms' in features:
            # Root mean square energy
            rms = librosa.feature.rms(y=debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
            get_feature_data('rms', rms, feature_data)

            # if 'spec_flux' in features:
            # spectral flux
            spec_flux = librosa.onset.onset_strength(y=debussy, sr=sr)
            get_feature_data('spec_flux', spec_flux, feature_data)

            # if 'spec_cont' in features:
            # spectral contrast
            S = np.abs(librosa.stft(debussy))
            spec_cont = librosa.feature.spectral_contrast(S=S, sr=sr)
            get_feature_data('spec_cont', spec_cont, feature_data)

            # if 'spec_cent' in features:
            # spectral centroid 
            spec_cent = librosa.feature.spectral_centroid(y=debussy, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
            get_feature_data('spec_cent', spec_cent, feature_data)

            # if 'spec_band' in features:
            # spectral bandwith
            spec_band = librosa.feature.spectral_bandwidth(y=debussy, sr=sr)
            get_feature_data('spec_band', spec_band, feature_data)

            

            # if 'roll_off' in features:
            # spectral roll off
            roll_off = librosa.feature.spectral_rolloff(y=debussy, sr=sr)[0]
            get_feature_data('roll_off', roll_off, feature_data)
            
            # if 'zcr' in features:
            # zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=debussy, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
            get_feature_data('zcr', zcr, feature_data)

            # Add mean, var, etc of each feature to csv

            to_append = f'{audio.replace(" ", "")} {tempo}'
    
            for f in features:
                for s in stats:
                    to_append += f' {feature_data[f,s]}'
            
            
            # if 'mfcc' in features:
            # mfcc
            mfcc = mfcss_feature_extractor(audio_path)
            for i in range(len(mfcc)):
            
                to_append += f' {np.mean(mfcc[i])}'
                to_append += f' {np.var(mfcc[i])}'
                to_append += f' {np.std(mfcc[i])}'
                to_append += f' {np.median(mfcc[i])}'
                to_append += f' {np.min(mfcc[i])}'
                to_append += f' {np.max(mfcc[i])}'


                

            # to_append += f' {label}'
            
            file = open(csv_file, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
    return csv_file

if __name__=="__main__": 



    

    # # write_to_csv(['tonnetz', 'spec_flux', 'spec_cont', 'spec_cent', 'roll_off', 'zcr'], 'model_feature_combo.csv')
    ## Split data into features and Labels
    data = pd.read_csv('model_feat_combo.csv')
    data = data.drop(['filename'],axis=1)
    data = data.drop(['tempo'],axis=1)
    
    

    x = np.array(data.iloc[:, :-1], dtype = float)
    y = np.array(data.iloc[:, -1])
    y = np.array(pd.get_dummies(y))


 

    ## Split into test and training data
    x_train, x_test, y_train, y_test = train_test_split(
                                                        #x[:,13:],
                                                        x, 
                                                        y, 
                                                        test_size=0.3
                                                    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)


  



    # print(predict('drake', 'test.csv'))
    # print(predict('71.wav'))
    # print(predict('93.wav'))
    # print(predict('209.wav'))




 
    model = models.load_model("music_model.keras")
    x_predict = model.predict(x_test_scaled) 
    pred = np.argmax(x_predict, axis = 1)
    print(y_test)
    print(pred)





    # tests = []
    # for x in range(3):
    #     tests.append(get_conv_model(x_train_scaled,x_test_scaled,y_train,y_test,2,64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy'))
    # print(tests)

    # model = get_conv_model(x_train_scaled,x_test_scaled,y_train,y_test,2,64, 64, 64, 3, 5, 7, 32, 'categorical_crossentropy')
    # model.save('music_model.keras')





