# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:30:04 2018

@author: JARD
"""

import pandas as pd
import os
import configparser
import platform
from lemonde import LemondeScrapping

def environment_variables():
    configParser = configparser.RawConfigParser() 
    if platform.system() in  ["Darwin", "Linux"]: # Mac or Linux
        configFilePath = os.environ["PROFILE"] + '/config_alexs.txt' # to check if PROFILE in os environ for Mac
    else:
        configFilePath = os.environ["USERPROFILE"] + '/config_alexs.txt'
    configParser.read(configFilePath)
    os.environ["DIR_PATH"] = configParser.get("config-Alexs", "project_path")

def main():
    
    ### crawl each competitor
    lemonde = LemondeScrapping()
    return lemonde

if __name__ == "__main__":
     environment_variables()
#     prices = main()