# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:49:49 2018

@author: User
"""

import os
import pandas as pd
import glob
import time
import shutil
import datetime
import numpy as  np

from utils_crawling import start, multiprocess_crawling


def get_stats_from_match(data_liste_matches, additionnal_path, i):
    
    data_liste_matches = data_liste_matches.reset_index(drop=True)
    driver, data = start()
    
    for j, href in enumerate(data_liste_matches.iloc[:,-1].tolist()):
        if "/QS" not in href:
            try:
                driver.get(href)
                div_desc = driver.find_element_by_xpath("//div[@id='completedMatchStats']/table/tbody").text.split("\n")
                time_desc = driver.find_element_by_xpath("//div[@class='scoring-section']/table").text.split("\n")

                data.append(data_liste_matches.iloc[j].tolist() + div_desc + time_desc)
                        
            except Exception:
                pass
    
    data = pd.DataFrame(data)   
    driver.close()
    
    data.to_csv(os.environ["DATA_PATH"] + additionnal_path + "/_tmp/extraction_brute_%i.csv"%i, index = False)
        

def get_list_href_matches(data_liste):
    
    data_liste = data_liste.reset_index(drop=True)
    driver, data = start()
    
    for i, compet_url in enumerate(data_liste["href"].tolist()):
        print(data_liste["tourney_name1"].iloc[i])
        driver.get(compet_url.replace("live-scores", "results"))
     
        try:
            header = driver.find_elements_by_xpath("//div[@class='tournery-results-container']/table/tbody")[0]
            master = header.find_element_by_tag_name("img").get_attribute('src')
            tourney_wrapper = header.text.split("\n") + [master.split("_")[1]]
           
            family = driver.find_elements_by_xpath("//div[@class='day-table-wrapper']/table/tbody")
            num_match= 1
            
            for parent in family[::-1]:
                elements = parent.find_elements_by_tag_name("tr")
                
                for el in elements[::-1]:
                    try:
                        score = el.find_element_by_class_name("day-table-score").get_attribute('innerHTML').split("\n")[3].replace("<sup>","(").replace("</sup>",")").split("\t")[0]
                        winner = el.find_elements_by_class_name("day-table-name")[0].text
                        loser = el.find_elements_by_class_name("day-table-name")[1].text
                        seed_w = el.find_elements_by_class_name("day-table-seed")[0].text
                        seed_l = el.find_elements_by_class_name("day-table-seed")[1].text
                        href_match_stats = el.find_element_by_class_name("day-table-score").find_element_by_tag_name("a").get_attribute('href')
                        
                    except Exception:
                        print(el.text)
                        winner = ""
                        loser = ""
                        href_match_stats = ""
                        score = "0-0 0-0"
                        seed_w = ""
                        seed_l= ""
                        
                    ### dont want qualifiers matches
                    if href_match_stats:
                        if  href_match_stats!="" and "/QS" not in href_match_stats:
                            data.append(data_liste.iloc[i].tolist() + [tourney_wrapper, winner, loser, score, num_match, seed_w, seed_l, href_match_stats])
                            num_match +=1
                    else:
                        data.append(data_liste.iloc[i].tolist() + [tourney_wrapper, winner, loser, score, num_match, seed_w, seed_l, ""])
                        num_match +=1
                        
        except Exception:
            print("No header for {0}".format(compet_url))
            pass
                
    data = pd.DataFrame(data)
    driver.close()
       
    return data


def get_list_href_competitions(url, latest):
    
    if "liste_tourney" in latest.keys():
        a= pd.DataFrame(latest["liste_tourney"], columns = ["id"])
        a["year"] = a["id"].apply(lambda x : int(x.split("/")[1]))
        liste_years = a["year"].unique()
    else:
        min_year = pd.to_datetime(latest["Date"]).year
        now = datetime.datetime.now()
        liste_years = range(min_year, now.year+1)
    
    driver, data = start()
    
    for year in liste_years:
        print("year {0}".format(year))
        driver.get(url + "/en/scores/results-archive?year=" + str(year))
        
        parent = driver.find_element_by_xpath("//div[@id='scoresResultsArchive']/table/tbody")
        elements = parent.find_elements_by_tag_name("tr")
        
        for el in elements:
            
            try:
                title = el.find_element_by_class_name("title-content").text
            except Exception:
                title = ""
                
            try:    
                details =  el.find_elements_by_class_name("tourney-details")
                surface_indoor = details[1].text
                cash = details[2].text
                href = details[4].find_element_by_tag_name("a").get_attribute('href')
            except Exception:
                surface_indoor = ""
                cash= ""
                href = ""
                
            if href != "":
                data.append([title, surface_indoor, cash, href])
    
    data = pd.DataFrame(data, columns = ["tournament", "surface_indoor", "prize", "href"])    
    data["Date"] = data["tournament"].apply(lambda x : pd.to_datetime(x.split(" ")[-1]))
    data = data.loc[data["Date"] >= pd.to_datetime(latest["Date"])]
    data["tourney_name1"] = data["href"].apply(lambda x :x.replace(url,"").split("/")[4])
    data["tourney_id"] = data["href"].apply(lambda x :x.replace(url,"").split("/")[5])
    
    driver.close()

    return data


def filter_data_liste(x, liste_to_keep):
    for el in liste_to_keep:
        if el in x:
            return True
        
    return False

def filter_already_id_tourney(x, already_tourney_id):
    
    x = x.split("/")[2] + "-" + x.split("/")[1]
    for el in already_tourney_id:
        if el in x:
            return False
        
    return True 

def extract_additionnal_data(latest):
    
    url = "https://www.atpworldtour.com"
    
    #### extract urls where tourney happened
    data_liste = get_list_href_competitions(url, latest)
    
    if "liste_tourney" in latest.keys():
        index = data_liste["href"].apply(lambda x : filter_data_liste(x.replace(url,""), latest["liste_tourney"]))
        data_liste = data_liste.loc[index]
        print(data_liste.shape)
        additionnal_path = "/brute_info/historical/correct_missing_values"
    
    if "already_tourney_id" in latest.keys():
        index = data_liste["href"].apply(lambda x : filter_already_id_tourney(x.replace(url,""), latest["already_tourney_id"]))
        data_liste = data_liste.loc[index]
        additionnal_path = "/clean_datasets/overall/updated/extracted"
        print(data_liste.shape)
    else:
        additionnal_path = "/clean_datasets/overall/updated/extracted"
        print(data_liste.shape)
    
    #### extract matches mesoscopic data (winner / loser and href for stats)
    data_liste_matches = get_list_href_matches(data_liste)
    data_liste_matches = data_liste_matches.loc[~pd.isnull(data_liste_matches.iloc[:,-1])]
    print(data_liste_matches.shape)

    ### extract microscopic statistics for each match 
    if os.path.isdir(os.environ["DATA_PATH"] + additionnal_path + "/_tmp"):
        shutil.rmtree(os.environ["DATA_PATH"] + additionnal_path + "/_tmp")
    
    try:    
        if not os.path.isdir(os.environ["DATA_PATH"] + additionnal_path +"/_tmp"):
            os.mkdir(os.environ["DATA_PATH"] + additionnal_path + "/_tmp")
    except Exception:
        if not os.path.isdir(os.environ["DATA_PATH"] + additionnal_path + "/_tmp"):
            os.mkdir(os.environ["DATA_PATH"] + additionnal_path + "/_tmp")
        pass

    ncore = 7
    multiprocess_crawling(get_stats_from_match, data_liste_matches, additionnal_path, ncore)
    
    last_time = time.time()
    while len(glob.glob(os.environ["DATA_PATH"] + additionnal_path + "/_tmp/*.csv")) !=ncore and time.time() - last_time < data_liste_matches.shape[0]*3:
        time.sleep(20)
        
    files = glob.glob(os.environ["DATA_PATH"] + additionnal_path + "/_tmp/*.csv")
    for i, f in enumerate(files):
        try:
            if i ==0:
                extra = pd.read_csv(f)
            else:
                extra = pd.concat([extra, pd.read_csv(f)], axis=0)
            
        except Exception:
             pass
     
    extra = extra.reset_index(drop=True)
    
    if "liste_tourney" in latest.keys():
        extra.to_csv(os.environ["DATA_PATH"] + "/brute_info/historical/correct_missing_values/missing_match_stats.csv", index = False)
    else:
        extra.to_csv(os.environ["DATA_PATH"] + "/clean_datasets/overall/updated/extracted/extraction_brute.csv", index = False)


if __name__ == "__main__":
    latest = {'Date': '2018-06-05',
              'already_tourney_id': np.array(['2018-339', '2018-451', '2018-891', '2018-301', '2018-338',
        '2018-580'])}
    extract_additionnal_data(latest)
    
    
    
    