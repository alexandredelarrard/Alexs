# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 09:03:36 2018

@author: User
"""

import os
from selenium import webdriver
import multiprocessing

def start():

    firefox_profile = webdriver.FirefoxProfile()
    firefox_profile.set_preference('permissions.default.stylesheet', 2)
    firefox_profile.set_preference('permissions.default.image', 2)
    firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
    firefox_profile.set_preference('disk-cache-size', 4096)
    firefox_profile.set_preference("http.response.timeout", 10)

    driver = webdriver.Firefox(firefox_profile=firefox_profile, log_path= os.environ["DIR_PATH"] + "/webdriver/geckodriver.log")#firefox_options=options, 
    driver.delete_all_cookies()
    driver.set_page_load_timeout(100)

    return driver


def multiprocess_crawling(function, data_liste_matches, additionnal_path, ncore):
    
    jobs = []
    nbr_core = ncore

    for i in range(nbr_core):
        
        sub_liste_refs = data_liste_matches[int(i*len(data_liste_matches)/nbr_core): int((i+1)*len(data_liste_matches)/nbr_core)]
        if i == nbr_core -1:
            sub_liste_refs = data_liste_matches[int(i*len(data_liste_matches)/nbr_core): ]
            
        p = multiprocessing.Process(target=function, args=(sub_liste_refs, additionnal_path,))
        jobs.append(p)
        p.start()