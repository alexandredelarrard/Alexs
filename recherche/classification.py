# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:59:15 2018

@author: User
"""

import numpy as np
import pandas as pd
import re
from gensim.models import FastText
import glob
import tqdm

from sklearn.model_selection import train_test_split
from keras.preprocessing import text, sequence
from keras import optimizers
from sklearn.metrics import log_loss, f1_score
from keras.models import Model
from keras.layers import Dense, Embedding, Input, CuDNNGRU, CuDNNLSTM, SpatialDropout1D
from keras.layers import LSTM, Bidirectional, Dropout, GRU, BatchNormalization, Conv2D, Conv1D, Activation, MaxPooling1D, Add, Flatten

from keras.models import model_from_json
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

import nltk
import tensorflow as tf
import itertools
import pickle

class Attention(Layer):
    def __init__(self, step_dim = 500,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)
        
        
    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def BidLstm(params, embedding_matrix):
    def top_2_categorical_accuracy(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2) 
    
    inp = Input(shape=(params['maxlen'], ))
    params['max_features'] = min(params['max_features'], embedding_matrix.shape[0])
    embedding_layer = Embedding(params['max_features'], params['embedding_dims'], 
                                weights=[embedding_matrix], trainable=False)(inp)
    
    x = Bidirectional(CuDNNGRU(params['lstm_size'], return_sequences=True))(embedding_layer)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(params['lstm_size'], return_sequences=True))(x)
    x = Attention(params['maxlen'])(x)
    x = Dense(params['dense_size'], activation="relu")(x)
    x = Dropout(params['dropout'])(x)
    
    x = Dense(len(classes), activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(clipvalue=1, clipnorm=1),
                  metrics=["accuracy"])

    print(model.summary()) 

    return model

lemmetizer = FrenchLefffLemmatizer()
def tokenize(text):
    text = re.sub(r'\S*@\S*\s?', ' ', text.strip(), flags=re.MULTILINE) # remove email
    text = re.sub(r'http\S+', ' ', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'www.\S+', ' ', text, flags=re.MULTILINE) # remove web addresses
    text = text.replace("\nLIRE AUSSI "," ").replace("\r\r\r\n",". ").replace("(Reuters)", " ").replace("€", " euros ").replace("$", " dollars ")
    text = text.lower().translate({ord(ch): None for ch in '0123456789'})
    text = text.translate({ord(ch): " " for ch in '©“’!"#$&\()*+\'-./:;<=>?@[\\]^_`{|}~,«»…\r'}) # lower + suppress numbers
    text = re.sub(' +',' ',text)
    liste_french= ["a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]
    text = " ".join([lemmetizer.lemmatize(lemmetizer.lemmatize(x, "v"), "n") for x in nltk.word_tokenize(text, language='french') if x not in liste_french]) 
    return text


def clean_sentence(X):
    print('Tokenizing data...')
    sentences = []
    for art in tqdm.tqdm(X["article"].tolist()):
        sentences.append(tokenize(art))
        
    try:
        y = X[classes].values
    except Exception:
        y= ""
        pass
    
    return sentences, y


def make_df(sentences, y, params):
    
    ##### ideea : only take words in both train and test comment text
    tok = text.Tokenizer(num_words= params['max_features'], char_level= False, lower=True)
    tok.fit_on_texts(sentences) 
    
    #### suppress words that I dont want to be used as encoded information
    phrases = tok.texts_to_sequences(sentences)
    x_train, x_test, y_train, y_test, index_train, index_test = train_test_split(phrases, y, range(len(sentences)), shuffle= True, test_size=0.2, random_state = 7666)
    
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
    
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=params['maxlen'])
    x_test = sequence.pad_sequences(x_test, maxlen=params['maxlen'])
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, x_test, y_train, y_test, tok, index_train, index_test


def create_embedding_matrix(model, params):
    embeddings_index = {}
    for word in tqdm.tqdm(model.wv.vocab.keys()):
        embeddings_index[word] = model[word]
    return embeddings_index
    

def prepare_embedding_fasttext(word_index, embeddings_index, params):

    words_not_found = []
    words_found = []
    len_words  = min(params['max_features'], len(word_index) + 1)
    embedding_matrix = np.zeros((len_words, params['embedding_dims']))
        
    print('Found %i word vectors.' % len(embeddings_index)) #### should be 2m words
    print('Found %i unique tokens. Len words = %i' %(len(word_index), len_words))
    j = 0
    for word, i in word_index.items():
        if len(words_found)>= len_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None and len(embedding_vector) > 0:
            embedding_matrix[j] = embedding_vector
            j +=1
            words_found.append(word)
        else:
            words_not_found.append(word)
            
    print("embedding matrix: " + str(embedding_matrix.shape))
    print("found %i words, %i words not found"%(len(words_found), len(words_not_found)))
    
    return embedding_matrix, words_not_found, words_found


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    
    best_loss = -1
    best_weights = None
    best_epoch = 0
    current_epoch = 0

    while current_epoch <= params["epochs"]:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(len(classes)):
            loss = log_loss(val_y[:, j].astype(np.float64), y_pred[:, j].astype(np.float64))
            total_loss += loss

        total_loss /= len(classes)
        real_accuracy(val_y, y_pred)
        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == params['patience']:
                break

    model.set_weights(best_weights)
    return model


def train_modl_wo_cv(X_train, Y_train, X_test, Y_test, params, embedding_matrix, i):
    
    fold_size = len(X_train) // max(params["n_splits"], 5)
    
    y_pred_test = np.ones((X_test.shape[0], len(classes)))
    
    for fold_id in range(0, params["n_splits"]):
        
        model = BidLstm(params, embedding_matrix)
        file_path = r"D:\articles_journaux\models\model_weights_%i.h5"%i
        
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X_train)

        train_x = np.concatenate([X_train[:fold_start], X_train[fold_end:]])
        train_y = np.concatenate([Y_train[:fold_start], Y_train[fold_end:]])
        val_x = X_train[fold_start:fold_end]
        val_y = Y_train[fold_start:fold_end]

        model = _train_model(model, params['batch_size'], train_x, train_y, val_x, val_y)
      
        model.save_weights(file_path)
        y_pred_test *= model.predict(X_test, batch_size = params['batch_size'])
          
    y_pred_test = (y_pred_test)**(1/params["n_splits"]) 
    total_loss = 0
    for j in range(len(classes)):
        loss = log_loss(Y_test[:, j], y_pred_test[:, j])
        total_loss += loss       
    real_accuracy(Y_test, y_pred_test)
    print( "_"*20 + "\n" +  " [Test {0}] loss : {1}".format(X_test.shape[0], total_loss))
            
    return model, X_test, Y_test, y_pred_test


def load_data():
    path = r"D:\articles_journaux\data\classes_article"
    files = glob.glob(path + "/*.csv")
    for i, file in enumerate(files):
        if i == 0 :
            X = pd.read_csv(file, sep= ",")
        else:
            X= pd.concat([X, pd.read_csv(file, sep= ",")], axis =0, sort= True)
            
    X = X.drop_duplicates("url")
    return X.reset_index(drop = True)


def read_new_articles(files, i):
    size = int((len(files))/6) + 1
    for i, file in enumerate(files[i*size: min(size, (i+1)*size)]):
        if i == 0 :
            articles = pd.read_csv(file, sep= ",")
        else:
            articles= pd.concat([articles, pd.read_csv(file, sep= ",")], axis =0)                                               
    return articles.reset_index(drop = True)


def real_accuracy(y, y_pred):
     y_p = np.where(y_pred >= 0.5, 1, 0)
     acc = []
     f1= []
     for i, k in enumerate(classes):
         print("[{0}] Accuracy : {1} , f1: {2}".format(k, ((y_p[:, i] == y[:,i])*1).sum()/len(y_p),f1_score(y[:,i], y_p[:,i])))
         acc.append(((y_p[:, i] == y[:,i])*1).sum()/len(y_p))
         f1.append(f1_score(y[:,i], y_p[:,i]))
     print(" TOTAL : accuracy : {0} f1_score : {1}".format(np.mean(acc),np.mean(f1)))


def predict_article(data, y_pred_test):
    yp = np.where(y_pred_test >= 0.4, 1, 0)
    data["prediction"] = [list(itertools.chain(*l.tolist())) for l in list(map(lambda x : np.argwhere(x == np.amax(x)), yp))]
    data["prediction"] = data["prediction"].apply(lambda x : [classes[a] for a in x])
    return data.reset_index(drop=True)
    

def filter_singificant_with_output(data, phrases, index):
    phrases = pd.DataFrame(phrases).loc[index].reset_index(drop=True)

    selected_index= data[["prediction", "classe"]].apply(lambda x : len(set(x[0]).intersection(eval(x[1]))) > 0 , axis = 1)
    data = data.reset_index(drop = True)
    return data[selected_index], list(phrases[selected_index]), np.array(data[selected_index][classes])
    

def filter_singificant(New, new_sentences, y_pred):
    
    yp = np.where(y_pred >= 0.75, 1, 0)
    yp = list(map(lambda x : sum(x), yp))
    index= np.where(yp >=  1, True, False)
    
    return New[index].reset_index(drop = True), list(pd.DataFrame(new_sentences).loc[index]), y_pred[index]

def semi_supervised():
    
    ### import data
    data = load_data()
    files = glob.glob(r"D:\articles_journaux\data\splits\*.csv")
    embedding_index = create_embedding_matrix(model, params)
    sentences, y = clean_sentence(data)
    
    ### predict train, suppress those not really significant 
    ### merge train + test + 10% random from full not in train test and loop
    for i in range(1):
        print(" ------>>  {0}th Round \n".format(i))
        X_t, X_te, y_t, y_te, tok, index_train, index_test = make_df(sentences, y, params)
        embedding_matrix, words_not_found, words_found = prepare_embedding_fasttext(tok.word_index, embedding_index, params)
        models, X_test, Y_test, y_pred_test = train_modl_wo_cv(X_t, y_t, X_te, y_te, params, embedding_matrix, i)
    
        Test = predict_article(data.loc[index_test], y_pred_test)
        Train = predict_article(data.loc[index_train], models.predict(X_t, batch_size = params["batch_size"]*2))
        
        #### keep only good articles
        data_train, sentence_train, y_tr = filter_singificant_with_output(Train, sentences, index_train)
        data_test, sentence_test, y_te = filter_singificant_with_output(Test, sentences, index_test)
        tt1 = read_new_articles(files, i)
        
        #### add new articles to the mix
        new_sentences, _ = clean_sentence(tt1)
        new_phrases = tok.texts_to_sequences(new_sentences)
        new_phrases = sequence.pad_sequences(new_phrases, maxlen=params['maxlen'])
        data_new, new_sentences, new_y = filter_singificant(tt1, new_sentences, models.predict(new_phrases, batch_size = params["batch_size"]*2))
        
        sentences = sentence_train  + sentence_test +  new_sentences # list 
        y = np.concatenate((y_tr, y_te, new_y), axis = 0) # array
        data = pd.concat([data_train, data_test, data_new], axis = 0).reset_index(drop =True) # dataframe
    
    return models, data


if __name__ == "__main__":
    
    global classes
    global params
    classes = ['Art', 'Bourse', 'Business', 'Economie', 'Education', 'Justice',
               'Planete', 'Politique', 'Sante', 'Science', 'Securite', 'Societe','Sport']
    
    model = FastText.load_fasttext_format(r'D:\articles_journaux\models\cc.fr.300.bin')
    params = {
              'max_features': 250000, 
              'maxlen': 500,
              'embedding_dims': 300,
              'batch_size': 256,
              'epochs': 30,
              'patience':  4,
              'n_splits' : 1,
              'dense_size' : 128, #128
              'lstm_size': 128,#128
              'dropout': 0.4,#0.4
              "classes" : classes
            }
    
    i = 0
    data = load_data()
    embedding_index = create_embedding_matrix(model, params)
    sentences, y = clean_sentence(data)
    X_t, X_te, y_t, y_te, tok, index_train, index_test = make_df(sentences, y, params)
    embedding_matrix, words_not_found, words_found = prepare_embedding_fasttext(tok.word_index, embedding_index, params)
    models, X_test, Y_test, y_pred_test = train_modl_wo_cv(X_t, y_t, X_te, y_te, params, embedding_matrix, i)

#    model_json = models.to_json()
#    with open(r"D:\articles_journaux\models\model.json", "w") as json_file:
#        json_file.write(model_json)
#        
#    import pickle
#    with open(r'C:\Users\User\Documents\Alexs\data\models\classification\tokenizer.pickle', 'wb') as handle:
#        pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    
#clas
#    Test = predict_article(data.loc[index_test], y_pred_test)
#    Train = predict_article(data.loc[index_train], models[0].predict(X_t, batch_size = params["batch_size"]*2))


    #### load model 
#    files = glob.glob(r"D:\articles_journaux\data\splits\*.csv")
#    json_file = open(r'C:\Users\User\Documents\Alexs\data\models\classification/model.json', 'r')
#    loaded_model_json = json_file.read()
#    json_file.close()
#    loaded_model = model_from_json(loaded_model_json, custom_objects={'Attention': Attention})
#    loaded_model.load_weights(r'C:\Users\User\Documents\Alexs\data\models\classification/model_weights_0.h5')
#    
#    with open(r'C:\Users\User\Documents\Alexs\data\models\classification/tokenizer.pickle', 'rb') as handle:
#            tok = pickle.load(handle)
#
#    ### recreate training data : 
#    full_data = read_new_articles(files, 0)
#    sentences, y = clean_sentence(full_data)
#    new_phrases = tok.texts_to_sequences(sentences)
#    new_phrases = sequence.pad_sequences(new_phrases, maxlen= 500)
#    y = loaded_model.predict(new_phrases, batch_size = params["batch_size"]*2)
#    
#    
#    yp = np.where(y >= 0.85, 1, 0)
#    yp = list(map(lambda x : sum(x), yp))
#    index= np.where(np.array(yp) >=  1, True, False)
#    
#    sub_data = full_data[index]
#    for i, col in enumerate(classes):
#        sub_data[col] = yp[index,i]
#        
#    sub_data = sub_data.reset_index(drop= True)
#    len_art = sub_data["article"].apply(lambda x : len(x)>500)
#    sub_data = sub_data[len_art].reset_index(drop= True)
#    sub_data.to_csv(r"D:\articles_journaux\data\classes_article\extraction_11112018.csv", index = False)

# [Test]   TOTAL : accuracy : 0.9826228172959556 f1_score : 0.8763522878383618
    
#[Art] Accuracy : 0.9863984098735757 , f1: 0.9184225117826449
#[Bourse] Accuracy : 0.9843934176161971 , f1: 0.8528466357940616
#[Business] Accuracy : 0.976801026186239 , f1: 0.8961217044837132
#[Economie] Accuracy : 0.9738426514433633 , f1: 0.873910258195694
#[Education] Accuracy : 0.9805336630688516 , f1: 0.8556864424930392
#[Justice] Accuracy : 0.9797189543994268 , f1: 0.8925619834710744
#[Planete] Accuracy : 0.9856472600365175 , f1: 0.851399856424982
#[Politique] Accuracy : 0.978852243048975 , f1: 0.9043237308516755
#[Sante] Accuracy : 0.9923556058890147 , f1: 0.868632707774799
#[Science] Accuracy : 0.984809439064414 , f1: 0.726516176011651
#[Securite] Accuracy : 0.9872824554510365 , f1: 0.9315417871916892
#[Societe] Accuracy : 0.9651697598631752 , f1: 0.8220148813038857
#[Sport] Accuracy : 0.9943259296923752 , f1: 0.9727992908980112
# TOTAL : accuracy : 0.9823177550487047 f1_score : 0.8743675358982248
#____________________
# [Test 173068] loss : 0.6097591659771397