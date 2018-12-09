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
        
        self.articles = articles.reset_index(drop=True)
        self.stemmer = FrenchStemmer()
        self.lemmetizer = FrenchLefffLemmatizer()
        self.liste_french= ["demain", "iii", "ii", "reuters", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
                            "beau", "encore", "tellement", "grand", "petit", "gros", "mince", "vieux", "vieille", "jamais", "toujours",     
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
        keep_index = self.articles["titre"].apply(lambda x : False if "L'essentiel de l'actu " in x else True)
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
        articles["article"] =  articles["titre"] + " " + articles["article"]
        articles["index"] = articles.index
        
        # =============================================================================
        #         #### 1) cluster duplicated articles and drop them
        # =============================================================================
        articles["deduplicate_cluster"] = self.step_clustering(articles, tresh_first_step = 0.75)
    
        # =============================================================================
        #         #### 2) cluster per subject and then topic
        # =============================================================================
        articles["granular_cluster"] = self.step_clustering(articles, tresh_first_step = 0.36)
        articles["cluster"] = self.step_clustering(articles, tresh_first_step = 0.23)

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
            
        article_words, tfs2 = self.weight_words(article_cluster, nwords= 100)
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
        
        self.articles["cluster"] = articles["cluster"].tolist()
        self.articles["cluster"] = articles["granular_cluster"].tolist()
        
        return cluster_words


    def step_clustering(self, articles, tresh_first_step):
        
        overall_cluster = {}
        length = articles["index"].shape[0]
        to_keep = articles["index"].tolist()
        
        while len(overall_cluster) != length:
            length = len(overall_cluster)
            sub_articles = articles.loc[to_keep].reset_index(drop =True)
            matrix_score, cluster = self.first_step_clustering(sub_articles, thresh=tresh_first_step)
            to_keep, mapping_cluster = self.select_center_cluster(sub_articles, matrix_score, cluster)
                                    
            if len(overall_cluster)> 0:
                for key, value in mapping_cluster.items():
                    tot = []
                    for x in value:
                        tot += overall_cluster[x]
                    mapping_cluster[key] = tot
            overall_cluster = mapping_cluster
        
        index_cluster = []
        for key, value in overall_cluster.items():
            for index_art in value:
                index_cluster.append([index_art, key])
     
        index_cluster = pd.DataFrame(index_cluster).sort_values(0)
        
        cluster_unique = index_cluster[1].value_counts()[index_cluster[1].value_counts() == 1].index
        index_cluster[1] = np.where(index_cluster[1].isin(cluster_unique), -1 , index_cluster[1])
        
        return index_cluster[1].tolist()
    

    def first_step_clustering(self, articles, thresh=0.375):
    
        matrix_score, clusters, tfs = self.new_intersect_cluster(articles, thresh)
        
        index_cluster = []
        for key, value in clusters.items():
            for index_art in value:
                index_cluster.append([index_art, key])
     
        index_cluster = pd.DataFrame(index_cluster).sort_values(0)
        articles["granular_cluster"] = index_cluster[1].tolist()
        cluster_unique = articles["granular_cluster"].value_counts()[articles["granular_cluster"].value_counts() == 1].index
       
        idx = articles["granular_cluster"].isin(cluster_unique)
        try:
            m = metrics.silhouette_score(tfs.toarray()[~idx], articles.loc[~idx]["granular_cluster"], metric='cosine')
            print("first step clustering {0}, {1}".format(m, len(cluster_unique)))
        except Exception:
            pass
        
        return matrix_score, clusters
    
    
    def match_general_cluster(self, cluster_words, thresh=0.37):
        
        if not os.path.isfile(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/general_cluster_words.json"):
            with open(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/general_cluster_words.json", "w") as f:
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
    
    
    def new_intersect_cluster(self, articles, thresh):
    
        article_words, tfs = self.weight_words(articles, nwords = 100)
     
        index_articles = list(range(len(article_words)))
        matrix_score = np.zeros((articles.shape[0], articles.shape[0]))
        liste_of_splits = [25, 50, 75, 100]
        for i in index_articles:
            for j in index_articles[i:]:
                score = 0
                for k in liste_of_splits:
                    dict_i = dict(article_words[i][:k]) 
                    dict_j = dict(article_words[j][:k]) 
                    intersect_words = list(set(dict_i.keys()).intersection(set(dict_j.keys())))
                    score += sum([dict_j[x] + dict_i[x] for x in intersect_words]) / (sum(dict_j.values()) + sum(dict_i.values())) #- 0.2*len(intersect_words) / len(article_words[j][k].keys())
                    
                matrix_score[i,j] = score/len(liste_of_splits)
                matrix_score[j,i] = score/len(liste_of_splits)
        
        cluster = {}
        positive_index= []
        
        j = 0
        articles_to_view = list(range(matrix_score.shape[0]))
        while len(articles_to_view) > 0:
            cluster[j]  = []
            new_list = [j]
            while len(new_list)> 0:
                p = new_list[0]
                positive_index = np.where(matrix_score[p] > thresh)
                new_p = articles.iloc[positive_index].index.tolist()
                new_p = list(set(new_p) - set(cluster[j]).intersection(set(new_p)))
                
                new_list = list(set(new_list + new_p))
                new_list.remove(p)
                cluster[j].append(p)
                articles_to_view.remove(p)
                
            if len(articles_to_view)> 0:
                j = articles_to_view[0]
                
        return matrix_score, cluster, tfs
    
    
    def weight_words(self, articles, nwords = 100):

        tfidf = TfidfVectorizer(stop_words = self.liste_french, preprocessor = self.tokenize, min_df = 2, max_df = 0.35, ngram_range=(1,2))
        tfs = tfidf.fit_transform(articles["article"].tolist())
        
        article_words = []
        features= tfidf.get_feature_names()
        for i, art in enumerate(articles["article"].tolist()):
            sorted_items= self.sort_coo(tfs[i].tocoo())
            article_words.append(self.extract_topn_from_vector(features, sorted_items, nwords))
    
        return article_words, tfs
    
    
    def select_center_cluster(self, sub_articles, matrix_score, clusters):
    
        mapping_cluster = {}
        mapping_rule = dict(sub_articles["index"])
        for key, value in clusters.items():
            if len(value)> 0:
                high_score = 0
                best = value[0]
                for element in value: 
                   score = sum(matrix_score[element][value])
                   if score > high_score:
                       best = element
                       high_score= score
                mapping_cluster[mapping_rule[best]] = [mapping_rule[x] for x in value] 
            else:
                mapping_cluster[mapping_rule[value[0]]]= mapping_rule[value[0]]

        return mapping_cluster.keys(), mapping_cluster
    
    
    def sort_coo(self, coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
    
    def extract_topn_from_vector(self, feature_names, sorted_items, topn=10):
        """get the feature names and tf-idf score of top n items"""
        
        #use only topn items from vector
        sorted_items = sorted_items[:topn]

        results= []
        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            results.append((feature_names[idx], round(score, 3)))
        
        return results