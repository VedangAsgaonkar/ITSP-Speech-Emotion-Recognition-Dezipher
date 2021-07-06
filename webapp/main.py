from subprocess import run, PIPE
import speech_recognition as sr
from flask import logging, Flask, render_template, request
from flask import Flask, render_template,request, redirect
import numpy as np
from werkzeug.utils import secure_filename
from subprocess import run, PIPE
import speech_recognition as sr
import librosa
import copy
import os


import tensorflow as tf
from tensorflow import keras

UPLOAD_FOLDER = 'downloads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def load_glove_embeddings(dim):
  #input: The dimensions of the embeddings to be loaded
  #output: A dictionary with every word and its vector

  GLOVE_DIM = 300 #the dimensions of the embeddings
  #loading the GLOVE embeddings
  input_path = 'glove/'
  glove_file =  'glove.6B.' + str(GLOVE_DIM) + 'd.txt'
  emb_dict = {} #dictionary that stores each vector indexed by its word
  glove = open(input_path+glove_file)
  for line in glove:
    word, vector = line.split(maxsplit=1)
    vector = np.fromstring(vector,'f',sep=" ")
    emb_dict[word] = vector
  glove.close()
  return emb_dict

text_data_path = "text/"

def load_train_data():
  #input: ideally the path should be input
  #output: training sentences and their labels

  train_text = open(text_data_path+"train.txt")
  angry_additional_text = open(text_data_path+"Emotion(angry).txt")

  #loading training data 
  train_sentences=[] #all sentences
  train_labels=[] #their labels
  train_as_str = "" #will hold all data as a string
  with train_text as f:
    train_as_str = f.read() #reading the training data
  train_as_str = train_as_str.lower()
  lines = train_as_str.split("\n") #each line has a sentence and a label, seperated with a semicolon
  for line in lines:
    lsen = line.split(";") #lsen has the sentence and its label
    if len(lsen) == 2 :
      train_sentences.append(lsen[0]) #storing the sentence
      train_labels.append(lsen[1]) #storing the label

  #now using the additional angry sentences. They are going to need further processing
  with angry_additional_text as f:
    train_as_str = f.read()
  train_as_str = train_as_str.lower()
  lines = train_as_str.split("\n")
  for line in lines:
    lsen = line.split(".")  #each line may have multiple sentences, hence split along dot
    train_sentences.extend(lsen) #adding all the sentences at the end
    train_labels.extend(["anger" for i in range(len(lsen))]) #adding "anger" the required number of times

  return train_sentences, train_labels


from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def create_vectorizer(sentence_data_for_vocab, num_most_freq_words=20000, max_word_length=24):
  #input: sentence_data_for_vocab: a list containing sentences from which the vectorizer create its vocabulary, which it maps to indices
  #       num_most_freq_words: int storing the number of most frequent words that will contribute to the vocabulary
  #       max_word_length: The length of the ouput int sentence from the vectorizer. Smaller sentences are padded, larger ones are truncated
  #output: vectorizer: the vectorizer which can be used to vectorize other sentences
  #        voc: a list of all words in descending order of their frequency of occurance
  #        word_index: dictionary storing words and their indices

  #vectorizer is used to convert sentences to a list of their word indices:
  vectorizer = TextVectorization(max_tokens=num_most_freq_words, output_sequence_length=max_word_length) #maximum number of words to be take= 20000, max length of sentence=24 words
  text_ds = tf.data.Dataset.from_tensor_slices(train_sentences).batch(128) #breaks into batches of 128 sentences. Not needed really, but okay
  vectorizer.adapt(text_ds)
  voc = vectorizer.get_vocabulary()
  word_index = dict(zip(voc, range(len(voc)))) #a dictionary with words and indices
  return vectorizer, voc, word_index

MFCC_F = keras.models.load_model('models/MFCC_Based_Detector_F.h5')
MFCC_M = keras.models.load_model('models/MFCC_Based_Detector_M.h5')
Mel_F = keras.models.load_model('models/Mel_Based_Detector_F.h5')
Mel_M = keras.models.load_model('models/Mel_Based_Detector_M.h5')
Words = keras.models.load_model('models/Words_Based_Detector.h5')

LENGTH_SENTENCE = 24
train_sentences, train_labels= load_train_data()
vectorizer, voc, word_index = create_vectorizer(train_sentences,20000,LENGTH_SENTENCE)


def frequency(audio_path):
    voice, samp = librosa.load(audio_path)
    mfcc = np.mean(librosa.feature.mfcc(y=voice,n_mfcc = 40).T, axis = 0)
    mel = np.mean(librosa.feature.melspectrogram(y=voice,n_mels = 256).T, axis = 0)
    mfcc = mfcc.reshape((1,40,1))
    mel = mel.reshape((1,256,1))
    

    mfcc_M_out = MFCC_M.predict(mfcc).reshape((3,1))
    mfcc_F_out = MFCC_F.predict(mfcc).reshape((3,1))
    

    mel_M_out = Mel_M.predict(mel).reshape((3,1))
    mel_F_out = Mel_F.predict(mel).reshape((3,1))
    
    return mel_M_out,mel_F_out,mfcc_M_out,mfcc_F_out

def first_second_max(arr):
  f = 0
  s = 0
  for c in arr:
    if c >= f:
      s = f
      f = c
    elif c>s:
      s = c 
  return f,s


numerical_labels = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "sadness": 3,    
    "love": 4,
    "surprise": 5,
}
numerical_list = ["anger","fear","joy","sadness","love","surprise"]
numerical_list_combined = ["anger","sad","happy"]




@app.route("/upload",methods=["GET","POST"])
def upload():
    transcript = ""
    emotions = {}
    final_result = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        f = request.files["file"]

        if f.filename == "":
            return redirect(request.url)
            
        filename = secure_filename(f.filename)
   
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.wav'))
   
        file_path = 'downloads/temp.wav' 

        r = sr.Recognizer()
        myvoice = sr.AudioFile(file_path)
        
        with myvoice as source:
            audio = r.record(source)
       
        text = r.recognize_google(audio, language = 'en-IN', show_all = False )
        print(text)
        
        x_train = vectorizer(np.array([[s] for s in [text]])).numpy()
        
        mel_M_out,mel_F_out,mfcc_M_out,mfcc_F_out = frequency(file_path)

        word_out = Words.predict(x_train).reshape((6,1))
        word_out_final = np.asarray([word_out[0][0],word_out[3][0],word_out[2][0]]).reshape((3,1))
        word_out_final = word_out_final/(np.sum(word_out_final))

        mel_first,mel_second = first_second_max(mel_M_out)
        mel_M_conf = mel_second/mel_first
        mel_first,mel_second = first_second_max(mel_F_out)
        mel_F_conf = mel_second/mel_first

        mfcc_first,mfcc_second = first_second_max(mfcc_M_out)
        mfcc_M_conf = mfcc_second/mfcc_first
        mfcc_first,mfcc_second = first_second_max(mfcc_F_out)
        mfcc_F_conf = mfcc_second/mfcc_first

        word_first,word_second = first_second_max(word_out_final)
        word_conf = word_second/word_first

        sound_features_M = (mel_M_out/mel_M_conf + mfcc_M_out/mfcc_M_conf )/(1/mel_M_conf + 1/mfcc_M_conf)
        sound_features_F = (mel_F_out/mel_F_conf + mfcc_F_out/mfcc_F_conf )/(1/mel_F_conf + 1/mfcc_F_conf)
        
        for r in range(3):
            if request.form["gender"]=="male":
                emotions['sound_features_X_' + numerical_list_combined[r]] = 1-sound_features_M[r,0]
            else:
                emotions['sound_features_X_' + numerical_list_combined[r]] = 1-sound_features_F[r,0]
            emotions['word_' + numerical_list_combined[r]] = 1-word_out_final[r,0]
        return render_template('upload.html',transcript= emotions, b= True)
        #end of if
        
    return render_template('upload.html',transcript= emotions, b= False)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/home")
def home1():
    return render_template('home.html')

@app.route("/contacts")
def contacts():
    return render_template('contacts.html')

@app.route("/audio")
def audio1():
    return render_template('index.html')

@app.route('/audio', methods=['POST'])
def audio():
    try:
      with open('/tmp/audio.wav', 'wb') as f:
          f.write(request.data)

      r = sr.Recognizer()
      myvoice = sr.AudioFile('/tmp/audio.wav')
      
      with myvoice as source:
          audio = r.record(source)
     
      text = r.recognize_google(audio, language = 'en-IN', show_all = False )
      
      print(text)
      
      x_train = vectorizer(np.array([[s] for s in [text]])).numpy()
      
      mel_M_out,mel_F_out,mfcc_M_out,mfcc_F_out = frequency('/tmp/audio.wav')

      word_out = Words.predict(x_train).reshape((6,1))
      word_out_final = np.asarray([word_out[0][0],word_out[3][0],word_out[2][0]]).reshape((3,1))
      word_out_final = word_out_final/(np.sum(word_out_final))
      print(word_out_final)


      mel_first,mel_second = first_second_max(mel_M_out)
      mel_M_conf = mel_second/mel_first
      mel_first,mel_second = first_second_max(mel_F_out)
      mel_F_conf = mel_second/mel_first

      mfcc_first,mfcc_second = first_second_max(mfcc_M_out)
      mfcc_M_conf = mfcc_second/mfcc_first
      mfcc_first,mfcc_second = first_second_max(mfcc_F_out)
      mfcc_F_conf = mfcc_second/mfcc_first

      word_first,word_second = first_second_max(word_out_final)
      word_conf = word_second/word_first

      sound_features_M = (mel_M_out/mel_M_conf + mfcc_M_out/mfcc_M_conf )/(1/mel_M_conf + 1/mfcc_M_conf)
      sound_features_F = (mel_F_out/mel_F_conf + mfcc_F_out/mfcc_F_conf )/(1/mel_F_conf + 1/mfcc_F_conf)
      emotions = {}
      for r in range(3):
        emotions['sound_features_M_' + numerical_list_combined[r]] = sound_features_M[r,0]
        emotions['sound_features_F_' + numerical_list_combined[r]] = sound_features_F[r,0]
        emotions['word_' + numerical_list_combined[r]] = word_out_final[r,0]
      
      print(emotions)

      proc = run(['ffprobe', '-of', 'default=noprint_wrappers=1', '/tmp/audio.wav'], text=True, stderr=PIPE)
      return  str(emotions)
      #end of try

    except:
      return "Audio Not Clear"


if __name__ == "__main__":
    #app.logger = logging.getLogger('audio-gui')
    app.run(debug=True)
