# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:12:47 2018

@author: User
"""


import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import glob
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from fastcluster import linkage
from scipy.spatial.distance import cdist, squareform
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier  
from sklearn import metrics
import hdbscan
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

def clean_articles2(x):

    liste_para = x[0].split("\r\n")
    end = re.sub("'\([^)]*\)", "", str(liste_para[-1]).replace(str(x[1]), "")).strip()
    article = "\r\n".join([x for x in liste_para[:-1] if x != ''] + [end])
    return article


lemmetizer = FrenchLefffLemmatizer()
def tokenize(text):
    text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
    text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = re.sub(r'www.\S+', '', text, flags=re.MULTILINE) # remove web addresses
    text = text.replace("\nLIRE AUSSI ","").replace("\n"," ").replace("(Reuters)", "")
    text = text.lower().translate({ord(ch): None for ch in '0123456789'})
    text = text.translate({ord(ch): " " for ch in '“’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»–…'}) # lower + suppress numbers
    tokens = nltk.word_tokenize(text, language='french')
    tokens = [lemmetizer.lemmatize(lemmetizer.lemmatize(lemmetizer.lemmatize(word, "v"), "n"), "a") for word in tokens]
    return tokens


def generate_text(tfidf, X, cluster, data):

    y = data['article_cluster'].map(lambda x: 1 if x == cluster else 0)   
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    clf = LogisticRegression(random_state = 0).fit(X_train, y_train)
    clf_d = DummyClassifier().fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    acc_d = clf_d.score(X_test, y_test)
    coef = clf.coef_.tolist()[0]
    w = tfidf.get_feature_names()
    coeff_df = pd.DataFrame({'words' : w, 'score' : coef})
    coeff_df = coeff_df.sort_values(['score', 'words'], ascending=[0, 1])
    coeff_df = coeff_df[:20]
    d = coeff_df.set_index('words')['score'].to_dict()
    
    words = pd.DataFrame(list(d.keys()))
    words["len"] = words[0].apply(lambda x: len(x))
    words = words.sort_values("len", ascending=False)
    keeping_words = []
    for word in words[0].tolist():
        if len([1 for x in keeping_words if word in x]) == 0:
            keeping_words.append(word)
    
    return keeping_words, d, acc, acc_d

def load_model():
    fname = get_tmpfile(r"C:\Users\User\Documents\Alexs\data\doc2vec\v2\doc2vec_articles_181030")
    model = Doc2Vec.load(fname) 
    return model


def similarity_plot(Y):
        
    fig = plt.figure(figsize=(15,15))
    
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(Y)
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
    
def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method,preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage


def similarity_within_cluster(model, data):
    
    similar = []
    for c in data["db_cluster"].value_counts().index:
        liste_articles = data.loc[data["db_cluster"] == c, "article"].tolist()
        if len(liste_articles)>1:
            
            X = []
            for x in liste_articles:
                X.append(model.infer_vector(x))
            Y = cdist(X,X, metric= "cosine")
            Y = np.round(Y, 6)
            print("Median similarity {1} for cluster {0}, shape {2}".format(c, Y.sum().sum()/(Y.shape[0]**2), len(liste_articles)))
            similar.append([c, Y.sum().sum()/(Y.shape[0]**2), max(Y)])
            
    return pd.DataFrame(similar)

files = glob.glob(r"C:\Users\User\Documents\Alexs\data\continuous_run\article\*.csv")
for i, file in enumerate(files):
    if i == 0 :
        articles = pd.read_csv(file, sep= "#")
    else:
        articles= pd.concat([articles, pd.read_csv(file, sep= "#")], axis =0)
                                                   
articles = articles.loc[articles["article"] !=""]
articles = articles.loc[~pd.isnull(articles["article"])]
articles = articles.loc[~pd.isnull(articles["restricted"])]
a = articles["article"].apply(lambda x : len(x))
articles = articles.loc[a>100]
articles["article"] = articles[["article", "auteur"]].apply(lambda x : clean_articles2(x), axis = 1)

liste_french= ["a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]

# =============================================================================
# #### hdbscan article
# =============================================================================
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words=liste_french, ngram_range=(1,4), max_features = 20000, min_df = 2)  

tfs = tfidf.fit_transform(articles["article"].tolist())
feature_names = tfidf.get_feature_names()

clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
cluster_labels = clusterer.fit_predict(tfs)
articles["article_cluster"] = cluster_labels

idx = articles["article_cluster"] !=-1
m = metrics.silhouette_score(tfs.toarray()[idx], articles.loc[idx]["article_cluster"], metric='cosine')
print("KPIS: silhouette {0}, shape {1}, -1 {2}".format(m, articles["article_cluster"].value_counts().shape[0], articles["article_cluster"].value_counts().iloc[0]))

#### analysis
articles["article_cluster"].value_counts()
articles["titre_cluster"].value_counts()
articles.loc[articles["article_cluster"] == 244, "titre"].tolist()

#### sigbificance words per cluster
# function for finding most significant words for each cluster

for i in articles["article_cluster"].value_counts().iloc[1:].index[-10:]:
    words, d, acc, acc_d = generate_text(tfidf, tfs, i, articles)
    print(" Cluster: {0} \n Words : {1}".format(i, words))
articles.loc[articles["article_cluster"] == 448, "titre"].tolist()

model = load_model()

X = []
for x in articles["article"].tolist():
    X.append(tokenize(x))
    
from gensim.similarities import MatrixSimilarity
    
Y = cdist(X,X, metric= "cosine")
Y = np.round(Y, 6)

articles = articles.reset_index(drop=True)
neg_articles = articles.loc[articles["article_cluster"] == -1]

for i in neg_articles.index:
    lui = []
    print(i)
    for j in articles["article"].index:
        if j !=i:
            lui.append(model.docvecs.similarity_unseen_docs(model, X[i], X[j]))

    if max(lui) > 0.65:
        rep = np.argmax(lui)
        print((i, rep))# , [articles.iloc[i]["article"], articles.iloc[rep]["article"]]

