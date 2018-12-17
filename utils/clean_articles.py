# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:14:25 2018

@author: User
"""
import re
import nltk
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import numpy as np
import itertools
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
lemmetizer = lemmetizer = FrenchLefffLemmatizer()

liste_french= ["demain", "iii", "ii", "reuters", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
                "beau", "encore", "tellement", "grand", "petit", "gros", "mince", "vieux", "vieille", "jamais", "toujours",     
                "fin", "afp", "déjà", "ok", "ca", "cas", "a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]


def error_handler(text):
    text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
    text = re.sub(r'\@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove tweeter names
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'\(Crédits :.+\)\r\n', ' ', text, flags=re.MULTILINE) # remove credits from start of article
    text = re.sub(r'\r\n.+\r\nVoir les réactions', '', text, flags=re.MULTILINE) # remove credits from start of article
    text = text.replace("/ REUTERS", "").replace("/ AFP", "").replace("%", " pourcent ").replace("TF1", " chaine television ")\
                .replace(" EI ", " Etat Islamique ").replace("1er","premier").replace(" CO2 "," dioxyde de carbone ").replace("n°"," numero ")\
                .replace(" 1ère "," premiere ").replace(' km² '," kilometre carre ").replace('m²'," metre carre ").replace('°C'," degre celcius ")\
                .replace("€"," euro ").replace("CAC40", " indice boursier ").replace("nyse", " new york stock exchange ").replace("macronie", " macron ")\
                .replace("NatIxis"," Natixis ").replace("Ligue1", " ligue un").replace("Ã§", "c").replace("Ã©", "é").replace("Ã¨", "è").replace('brexiters', "brexit ") \
                .replace(" \x92", " ").replace(" \xad", " ").replace("lrem", "la republique en marche").replace("LREM", "la republique en marche").replace('lfi', "la France insoumise").replace("LaREM", "la republique en marche")\
                .replace("gafa", " Google Apple Facebook Amazone ").replace("PS2", " console de jeu ").replace('Parcoursup'," parcours supérieur").replace("FESF", " Fonds européen de stabilité financière ")\
                .replace("OSDH", "Observatoire syrien des droits de l'homme")
    s = text.split("(Reuters) - ",1)
    if len(s)>1:
        text = s[1]
    s = text.split(" - ",1)
    if len(s[0]) < 35:
        text = s[1]
    text = re.sub(r'\r\npar .+\r\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\r\n\(.+\)', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nÀ LIRE AUSSI\r\n.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE AUSSI\r\n.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE AUSSI >>.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\n» LIRE AUSSI -.+\r\n', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLE FIGARO. -.+ - ', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r'www.\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'\r\n» Vous pouvez également suivre.+.', '', text, flags=re.MULTILINE) 
    text = re.sub(r'\r\nLIRE NOTRE DOSSIER COMPLET\r\n.+\r\n', '', text, flags=re.MULTILINE) 
    
    text = text.replace("\r\nLIRE AUSSI :\r\n»", "")
    text = text.replace("(Reuters)", "").replace("Article réservé aux abonnés", " ")
    return text
    

def tokenize(text):
    text = error_handler(text)
    text = text.translate({ord(ch): None for ch in '0123456789'})
    text = text.translate({ord(ch): " " for ch in '-•“’!"#$%&()*+,./:;<=>?@[\\]^_`{|}~«»–…‘'}) # lower + suppress numbers
    text = re.sub(r' +', ' ', text, flags=re.MULTILINE) # remove autor name
    text = re.sub(r' \b[a-zA-Z]\b ', ' ', text, flags=re.MULTILINE) ### mono letters word
    tokens = nltk.word_tokenize(text.lower(), language='french')  
    tokens = " ".join([lemmetizer.lemmatize(lemmetizer.lemmatize(lemmetizer.lemmatize(word, "v"), "n"), "a") for word in tokens])
    return tokens


def classification_tokenize(text):
    text = error_handler(text)
    text = text.translate({ord(ch): '0' for ch in '0123456789'})
    text = text.translate({ord(ch): " " for ch in "🏆-•“’\”!\"#$&()*+,./:;<=>?@[\\]^_`{|}~«»–…‘'"})
    text = re.sub(r' +', ' ', text, flags=re.MULTILINE) 
    text = re.sub(r' \b[a-zA-Z]\b ', ' ', text, flags=re.MULTILINE) ### mono letters word
    
    liste_words = []
    for x in text.split():
        if x.isupper():
            liste_words.append(x.lower())
        else:
            liste_words.append(x)
    text = " ".join([x for x in liste_words if x not in liste_french])
    return text


def load_doc2vec_model():
    fname = get_tmpfile(os.path["DIR_PATH"] + "/data/doc2vec/v2/doc2vec_articles_181030")
    model = Doc2Vec.load(fname) 
    return model


def from_output_to_classe(y, classes):
    yp = np.where(y >= 0.5, 1, 0)
    
#    yp1 = np.zeros((y.shape[0], y.shape[1]))
#    maxi = np.argmax(y,axis=1)
#    for index in range(y.shape[0]):
#        yp1[index, maxi[index]] = 1
#    
#    index = np.sum(yp, axis=1) ==0
#    yp[index, :] = yp1[index, :]
    
    pred = [list(itertools.chain(*l.tolist())) for l in list(map(lambda x : np.argwhere(x == np.amax(x)), yp))]
    
    predictions = []
    for a in pred:
        sub = []
        if len(a) < 4:
            for id_ in a:
                sub.append(classes[id_])
            predictions.append(sub)
        else:
            predictions.append(None)

    return predictions