# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import hdbscan
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import json
from sklearn import metrics
import numpy as np
import os
from datetime import datetime
import tqdm

class ClusteringArticles(object):
    
    def __init__(self, articles):
        self.articles = articles
        self.stemmer = FrenchStemmer()
        self.lemmetizer = FrenchLefffLemmatizer()
        self.liste_french= ["demain", "iii", "ii", "reuters", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
#                            "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre",
                            "fin", "afp", "déjà", "ok", "ca", "cas", "a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]
        
        
    def main_article_clustering(self):
        self.clean_articles()
        cluster_words = self.clustering_Tf_Itf()
#        self.match_general_cluster(cluster_words)
        return self.articles
        
    
    def clean_articles(self):
        
        def clean_articles2(x):
            liste_para = x[0].split("\r\n")
            end = re.sub("'\([^)]*\)", "", str(liste_para[-1]).replace(str(x[1]), "")).strip()
            article = "\r\n".join([x for x in liste_para[:-1] if x != ''] + [end])
            return article
    
        self.articles = self.articles.loc[~pd.isnull(self.articles["article"])]
        self.articles = self.articles.loc[~pd.isnull(self.articles["restricted"])]
        self.articles = self.articles.loc[self.articles["article"].apply(lambda x : len(x)) > 750]
        keep_index = self.articles["article"].apply(lambda x : False if "L'essentiel de l'actu " in x else True)
        self.articles = self.articles.loc[keep_index]
        self.articles["article"] = self.articles[["article", "auteur"]].apply(lambda x : clean_articles2(x), axis = 1)
        
    
    def tokenize(self, text):
        text = text.replace("'"," ")
        text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
        text = re.sub(r'\@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove tweeter names
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
        text = re.sub(r'\(Crédits :.+\)\r\n', ' ', text, flags=re.MULTILINE) # remove credits from start of article
        text = re.sub(r'\r\n.+\r\nVoir les réactions', '', text, flags=re.MULTILINE) # remove credits from start of article
        text = text.replace("/ REUTERS", "").replace("/ AFP", "")
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
        text = text.translate({ord(ch): None for ch in '0123456789'})
        text = text.translate({ord(ch): " " for ch in '-•“’!"#$%&()*+,./:;<=>?@[\\]^_`{|}~«»–…‘'}) # lower + suppress numbers
        text = re.sub(r' +', ' ', text, flags=re.MULTILINE) # remove autor name
        text = re.sub(r' \b[a-zA-Z]\b ', ' ', text, flags=re.MULTILINE) ### mono letters word
        tokens = nltk.word_tokenize(text.lower(), language='french')
    
        tokens = " ".join([self.lemmetizer.lemmatize(self.lemmetizer.lemmatize(word, "v"), "n") for word in tokens])
        return tokens
    
    
    def clustering_Tf_Itf(self, thresh=0.6, nwords = 100):
        '''
         Home made clustering method:
             - get nwords most important words per document (after tf idf)
             - Group articles having at least thresh of common weights (% of importance in common between articles)
             - If one group then cluster = -1
        '''
        
        articles = self.articles.copy()
        
        # =============================================================================
        #         #### 1) cluster articles close in words
        # =============================================================================
        clusters, tfs = self.intersect_cluster(articles, nwords= 70, thresh = 0.55)
        index_cluster = []
        for key, value in clusters.items():
            for index_art in value:
                index_cluster.append([index_art, key])
     
        index_cluster = pd.DataFrame(index_cluster).sort_values(0)
        articles["article_cluster"] = index_cluster[1].tolist()
        cluster_unique = articles["article_cluster"].value_counts()[articles["article_cluster"].value_counts() == 1].index
       
        idx = articles["article_cluster"].isin(cluster_unique)
        m = metrics.silhouette_score(tfs.toarray()[~idx], articles.loc[~idx]["article_cluster"], metric='cosine')
        print("first step clustering {0}, {1}".format(m, len(cluster_unique)))

        
        # =============================================================================
        #         #### 2) cluster clusters 
        # =============================================================================
        articles["article"] =  articles["titre"] + " " + articles["titre"] + " " + articles["titre"] + " " + articles["article"]
        article_cluster = {}
        for cluster in articles["article_cluster"].value_counts().sort_index().index:
            sub_articles = articles.loc[articles["article_cluster"] == cluster, "article"].tolist()
            a = ""
            for art in sub_articles:
                a += " " + art
            article_cluster[cluster] = a
        article_cluster= pd.DataFrame.from_dict(article_cluster, orient = "index").sort_index()
        article_cluster.columns= ["article"]
            
        clusters2, tfs2 = self.intersect_cluster(article_cluster, 70, 0.37)
        articles["cluster"] = 0
        for key, value in clusters2.items():
            value_cluster = article_cluster.iloc[value].index.tolist()
            articles.loc[articles["article_cluster"].isin(value_cluster), "cluster"] = articles.loc[articles["article_cluster"].isin(value_cluster), "article_cluster"].iloc[0]
        
        # =============================================================================
        #         #### 3) get main words from cluster to merge with past clusters
        # =============================================================================
        article_cluster = {}
        liste_cluster = []
        for cluster in articles["cluster"].value_counts().sort_index().index:
            sub_articles = articles.loc[articles["cluster"] == cluster, "article"].tolist()
            a = ""
            for art in sub_articles:
                a += " " + art
            liste_cluster.append(cluster)
            article_cluster[cluster] = a
        article_cluster= pd.DataFrame.from_dict(article_cluster, orient = "index").sort_index()
        article_cluster.columns= ["article"]
            
        liste_words, article_words, tfs2 = self.weight_words(article_cluster, nwords= 70)
        cluster_words = {}
        for i, words in enumerate(article_words):
            cluster_words[liste_cluster[i]] = words
        
        with open(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/dayly_cluster/{0}.json".format(datetime.now().strftime("%Y-%m-%d")), "w") as f:
            json.dump(cluster_words, f, ensure_ascii=False, indent=2)
        
        # =============================================================================
        #         ### 4) finish : lonely cluster into -1
        # =============================================================================
        cluster_unique = articles["cluster"].value_counts()[articles["cluster"].value_counts() == 1].index
        articles["cluster"] = np.where(articles["cluster"].isin(cluster_unique), -1 , articles["cluster"])
        
        idx = articles["cluster"] != -1
        m = metrics.silhouette_score(tfs.toarray()[idx], articles.loc[idx]["cluster"], metric='cosine')
        minus =  articles["cluster"].value_counts().iloc[0]
        print("clustering {0} , {1}".format(m, minus))
        
        self.articles["cluster"] = articles["cluster"].tolist()
        self.articles["granular_cluster"] = articles["article_cluster"].tolist()
        
        return cluster_words
        
        
    def match_general_cluster(self, cluster_words, thresh=0.37):
        
        if not os.path.isfile(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/general_cluster_words.json"):
            with open(os.environ["DIR_PATH"] + "/data/continuous_run/general_cluster_words.json", "w") as f:
                json.dump(cluster_words, f, ensure_ascii=False, indent=2)
            return 0
            
        with open(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/general_cluster_words.json", "r") as read_file:
            general_cluster_words = json.load(read_file)
            
        max_cluster = max([int(x) for x in general_cluster_words.keys()])
        
        rematch_cluster = {}
        additionnal_dico = {}
        for new_cluster, new_words in tqdm.tqdm(cluster_words.items()):
            max_score = 0
            for cluster, words in general_cluster_words.items():
                intersect_words = list(set(new_words.keys()).intersection(set(words.keys())))
                score = sum([new_words[x] + words[x] for x in intersect_words]) *2 / (sum(new_words.values()) + sum(words.values()))
                if score >= thresh: 
                    max_score = score
                    rematch_cluster[new_cluster] = cluster
                
            ### if not in treshold, create new cluster
            if max_score < thresh:
                additionnal_dico[str(max_cluster + 1)] = new_words
                rematch_cluster[new_cluster] = str(max_cluster + 1)
                max_cluster +=1
                
        general_cluster_words.update(additionnal_dico)
        with open(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/general_cluster_words.json", "w") as f:
                json.dump(general_cluster_words, f, ensure_ascii=False, indent=2)
         
        for key, value in rematch_cluster.items():
            self.articles.loc[self.articles["cluster"] == key, "cluster"] = value
        
    
    def intersect_cluster(self, articles, nwords, thresh):
    
        liste_words, article_words, tfs = self.weight_words(articles, nwords)
        
        clusters = {}    
        index_articles = list(range(len(article_words)))
        cluster = 0
        while len(index_articles)> 1:
            j = index_articles[0]
            clusters[str(cluster)] = []
            for k in index_articles:
                intersect_words = list(set(article_words[k].keys()).intersection(set(article_words[j].keys())))
                score = sum([article_words[j][x] + article_words[k][x] for x in intersect_words]) *2 / (sum(article_words[j].values()) + sum(article_words[k].values()))
                if score >= thresh: 
                    clusters[str(cluster)].append(k)
                    index_articles.remove(k)
            cluster +=1
            
        if len(index_articles) == 1:
            clusters[str(cluster)] =  [index_articles[0]]
                
        return clusters, tfs
    
    
    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
    
    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        
        #use only topn items from vector
        sorted_items = sorted_items[:topn]
     
        score_vals = []
        feature_vals = []
        
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            
            #keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
     
        #create a tuples of feature,score
        #results = zip(feature_vals,score_vals)
        results= {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        
        return results
    
    
    def weight_words(self, articles, nwords = 40):

        tfidf = TfidfVectorizer(stop_words = self.liste_french, preprocessor = self.tokenize, min_df = 2, max_df = 0.35, ngram_range=(1,2))
        tfs = tfidf.fit_transform(articles["article"].tolist())
        
        liste_words = []
        article_words = []
        for i, art in enumerate(articles["article"].tolist()):
            sorted_items=self.sort_coo(tfs[i].tocoo())
            article_words.append(self.extract_topn_from_vector(tfidf.get_feature_names(), sorted_items, nwords))
            liste_words += article_words[i].keys()
    
        return liste_words, article_words, tfs
    
    
    def db_scan_on_top_k(self, articles):
    
        liste_words, article_words, tfs = self.weight_words(articles, 100)
        liste_words = list(set(liste_words))
        
        index_col_keep = []
        for word in liste_words:
            index_col_keep.append(liste_words.index(word))
            
        tfs = tfs[:,index_col_keep]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(tfs.toarray())
        articles["article_cluster"] = cluster_labels
        
        idx = articles["article_cluster"] !=-1
        m = metrics.silhouette_score(tfs.toarray()[idx], articles.loc[idx]["article_cluster"], metric='cosine')
        print("KPIS: silhouette {0}, shape {1}, -1 {2}".format(m, articles["article_cluster"].value_counts().shape[0], articles["article_cluster"].value_counts().iloc[0]))
    
        return articles, tfs
    
    
    def clustering_Tf_ItfV1(self):
        tfidf = TfidfVectorizer(preprocessor=self.tokenize, stop_words=self.liste_french, ngram_range=(1,2), 
                               max_features = 2000, max_df= 0.25, min_df = 3)
        
        tfs = tfidf.fit_transform(self.articles["article"].tolist()) 
        # HDBSCAN / -1 = no cluster
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(tfs.toarray())
        self.articles["cluster"] = cluster_labels
        
        ### get most used words per cluster detected
        self.articles["main_words"] = ""
        for i in self.articles["cluster"].value_counts().iloc[1:].index:
            try:
                keeping_words, d = self.generate_text(tfidf, tfs, i, self.articles)
                self.articles.loc[self.articles["cluster"] == i, "main_words"] = keeping_words
            except Exception:
                self.articles.loc[self.articles["cluster"] == i, "main_words"] = ""
                pass

        ### silhouette
        idx = self.articles["cluster"] !=-1
        m = metrics.silhouette_score(tfs.toarray()[idx], self.articles.loc[idx]["cluster"], metric='cosine')
        print("KPIS: silhouette {0}, shape {1}, -1 {2}".format(m, self.articles["cluster"].value_counts().shape[0], self.articles["cluster"].value_counts().iloc[0]))
    
    