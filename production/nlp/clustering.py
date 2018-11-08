# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
import hdbscan
from nltk.stem.snowball import FrenchStemmer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class ClusteringArticles(object):
    
    def __init__(self, articles):
        self.articles = articles
        self.stemmer = FrenchStemmer()
        self.lemmetizer = FrenchLefffLemmatizer()
        
        
    def main_article_clustering(self):
        self.clean_articles()
        self.clustering_Tf_Itf()
        return self.articles
        
    
    def clean_articles(self):
        
        def clean_articles2(x):
            liste_para = x[0].split("\r\n")
            end = re.sub("'\([^)]*\)", "", str(liste_para[-1]).replace(str(x[1]), "")).strip()
            article = "\r\n".join([x for x in liste_para[:-1] if x != ''] + [end])
            return article
    
        self.articles = self.articles.loc[~pd.isnull(self.articles["article"])]
        self.articles = self.articles.loc[~pd.isnull(self.articles["restricted"])]
        a = self.articles["article"].apply(lambda x : len(x))
        self.articles = self.articles.loc[a>100]
        self.articles["article"] = self.articles[["article", "auteur"]].apply(lambda x : clean_articles2(x), axis = 1)
        
    
    def tokenize(self, text):
        text = re.sub(r'\S*@\S*\s?', '', text.strip(), flags=re.MULTILINE) # remove email
        text = re.sub(r'http\S+', '', text, flags=re.MULTILINE) # remove web addresses
        text = re.sub(r'\(Crédits :.+\)', '', text, flags=re.MULTILINE) # remove credits from start of article
        text = re.sub(r'www.\S+', '', text, flags=re.MULTILINE) # remove web addresses
        text = text.replace("\nLIRE AUSSI ","").replace("(Reuters)", "")
        text = text.lower().translate({ord(ch): None for ch in '0123456789'})
        text = text.translate({ord(ch): " " for ch in '“’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»–…'}) # lower + suppress numbers
        tokens = nltk.word_tokenize(text, language='french')
        tokens = [self.lemmetizer.lemmatize(self.lemmetizer.lemmatize(self.lemmetizer.lemmatize(word, "v"), "n"), "a") for word in tokens]
        return tokens
    
    
    def clustering_Tf_Itf(self):
        liste_french= ["a","abord","absolument","afin","ah","ai","aie","aient","aies","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aucuns","aujourd","aujourd'hui","aupres","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","bon","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","celà","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","devrait","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","dos","douze","douzième","dring","droite","du","duquel","durant","dès","début","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","essai","est","et","etant","etc","etre","eu","eue","eues","euh","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eux","eux-mêmes","exactement","excepté","extenso","exterieur","eûmes","eût","eûtes","f","fais","faisaient","faisant","fait","faites","façon","feront","fi","flac","floc","fois","font","force","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fûmes","fût","fûtes","g","gens","h","ha","haut","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","ici","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","mine","minimale","moi","moi-meme","moi-même","moindres","moins","mon","mot","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","nommés","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nouveaux","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parole","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","personnes","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","pièce","plein","plouf","plupart","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soient","sois","soit","soixante","sommes","son","sont","sous","souvent","soyez","soyons","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","sujet","superpose","sur","surtout","t","ta","tac","tandis","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","valeur","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voie","voient","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","état","étiez","étions","été","étée","étées","étés","êtes","être","ô"]
        pca = PCA(n_components=1000, svd_solver='randomized')
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, stop_words=liste_french, ngram_range=(1,2), 
                               max_features = 1000, max_df=0.3, min_df = 3, lowercase = False)
        tfs = tfidf.fit_transform(self.articles["article"].tolist())
        pca.fit(tfs.toarray()) 
        # HDBSCAN / -1 = no cluster
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(pca.transform(tfs.toarray()))
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

    def generate_text(self, tfidf, X, cluster, data):

        y = data['cluster'].map(lambda x: 1 if x == cluster else 0)   
        
        clf = LogisticRegression(random_state = 0).fit(X, y)
        coef = clf.coef_.tolist()[0]
        w = tfidf.get_feature_names()
        coeff_df = pd.DataFrame({'words' : w, 'score' : coef})
        coeff_df = coeff_df.sort_values(['score', 'words'], ascending=[0, 1])
        coeff_df = coeff_df[:20]
        d = coeff_df.set_index('words')['score'].to_dict()
        
        words = pd.DataFrame(list(d.keys()))
        words["len"] = words[0].apply(lambda x: len(x))
        words = words.sort_values("len", ascending=False)
        keeping_words = ""
        for word in words[0].tolist():
            if len([1 for x in keeping_words if word in x]) == 0:
                keeping_words += ", " + word
        return keeping_words[2:], d
    