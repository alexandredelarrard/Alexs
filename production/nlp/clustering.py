# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:56:14 2018

@author: User
"""

import pandas as pd
import re
import json
from sklearn import metrics
import numpy as np
import os
from datetime import datetime
import tqdm

from utils.extract_words import weight_words, get_top_k_words_cluster

class ClusteringArticles(object):
    
    def __init__(self, articles):
        self.articles = articles

    def main_article_clustering(self):
        self.clean_articles()
        clusters = self.clustering_Tf_Itf()
#        self.match_general_cluster(cluster_words)
        return self.articles, clusters
        
    
    def clean_articles(self):
        
        def clean_articles2(x):
            liste_para = x[0].split("\r\n")
            end = re.sub("'\([^)]*\)", "", str(liste_para[-1]).replace(str(x[1]), "")).strip()
            article = "\r\n".join([x for x in liste_para[:-1] if x != ''] + [end])
            return article
    
        self.articles = self.articles.loc[~pd.isnull(self.articles["article"])]
        self.articles = self.articles.loc[~pd.isnull(self.articles["restricted"])]
        self.articles = self.articles.loc[self.articles["article"].apply(lambda x : len(x)) > 750]
        keep_index = self.articles["titre"].apply(lambda x : False if "l'essentiel de l'actu " in x.lower() else True)
        self.articles = self.articles.loc[keep_index]
        self.articles["article"] = self.articles[["article", "auteur"]].apply(lambda x : clean_articles2(x), axis = 1)
        self.articles = self.articles.reset_index(drop=True)
    
    
    def clustering_Tf_Itf(self):
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
#        articles["deduplicate_cluster"] = self.step_clustering(articles, tresh_first_step = 0.75)
    
        # =============================================================================
        #         #### 2) cluster per subject and then topic
        #               solo_cluster = True : possible d avoir un cluster avec un seul article, 
        #               sinon ils sont tous regroupes sous la valeur -1   
        # =============================================================================
        articles["granular_cluster"] = self.step_clustering(articles, tresh_first_step = 0.365, solo_cluster= False)
        articles["cluster"] = self.step_clustering(articles, tresh_first_step = 0.23, solo_cluster= True)
        
        self.articles["cluster"] = articles["cluster"].astype(str).tolist()
        self.articles["granular_cluster"] = articles["granular_cluster"].tolist()

        # =============================================================================
        #         ### Create clusters database : cluster, mot_cles, titre, sujet/theme, 
        # =============================================================================
        total = {}
        titres = []
        for item in articles["cluster"].value_counts().index:
            sub_cluster = articles.loc[articles["cluster"] == item]
            titres.append(sub_cluster["titre"].iloc[0])
            cluster_words, tfs3 = weight_words(sub_cluster, nwords = 100, token= False)
            total[item] = {"mot_cles" : get_top_k_words_cluster(cluster_words)[-5:].index.tolist(),
                           "cluster" : str(item),
                           "titre" : sub_cluster["titre"].iloc[0],
                           "nbr_articles" : sub_cluster.shape[0]}
        return total


    def step_clustering(self, articles, tresh_first_step, solo_cluster = True):
        
        overall_cluster = {}
        length = articles["index"].shape[0]
        to_keep = articles["index"].tolist()
        
        while len(overall_cluster) != length:
            length = len(overall_cluster)
            sub_articles = articles.loc[to_keep].reset_index(drop =True)
            matrix_score, cluster, tfs = self.new_intersect_cluster(sub_articles, tresh_first_step)
            to_keep, mapping_cluster   = self.select_center_cluster(sub_articles, matrix_score, cluster)
                                    
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
        
        if not solo_cluster:
            cluster_unique = index_cluster[1].value_counts()[index_cluster[1].value_counts() == 1].index
            index_cluster[1] = np.where(index_cluster[1].isin(cluster_unique), -1 , index_cluster[1])
        
        return index_cluster[1].tolist()
    
    
    def new_intersect_cluster(self, articles, thresh):
    
        """
            this function cluster all articles based on a treshold. 
            The higher the similarity between articles , the higher the score, the higher the propensity to reach the treshold 
        """
        
        article_words, tfs = weight_words(articles, nwords = 100)
     
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
    
    
    def select_center_cluster(self, sub_articles, matrix_score, cluster):
        """
             - function gives the article that is the most correlated with every articles in the same cluster
             It is used in order to merge clusters between them as being closest distance between center of clusters
        """
    
        mapping_cluster = {}
        mapping_rule = dict(sub_articles["index"])
        for key, value in cluster.items():
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

# =============================================================================
#     functions for time clustering
# =============================================================================
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
    
    
    def time_clustering(self, articles):
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
            
        article_words, tfs2 = weight_words(article_cluster, nwords= 100)
        cluster_words = {}
        for i, words in enumerate(article_words):
            cluster_words[liste_cluster[i]] = words
        
        with open(os.environ["DIR_PATH"] + "/data/continuous_run/clusters/dayly_cluster/{0}.json".format(datetime.now().strftime("%Y-%m-%d")), "w") as f:
            json.dump(cluster_words, f, ensure_ascii=False, indent=2)
        