#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import os
import re
from collections import defaultdict
from sys import path

path.append('../PyHTK/')
import HTK
from HTK import HTKFile


def load_coefficients(filepath, filenameList = None, speakersId = None,
                      speakerIdRegex = "(.+)_"):
    coefficientsDict = defaultdict(list)
    # If the filenameList is not provided, it's used all files *.out
    if filenameList == None:
        filenameList = find_all_files(filepath, ".out");
    #speakerIdRegex is used to cluster the coefficients by the speaker ID         
    for filename in sorted(filenameList):
        match = re.search(speakerIdRegex,filename)
        if match is None:
            continue
        speaker = match.group(1)
        if speakersId is not None:
            if speaker not in speakersId:
                continue
        coefficientsDict[speaker].append(numpy.loadtxt(filepath 
                                         +filename, delimiter=","))            
    return coefficientsDict

def find_all_files(filepath, fileExtension):
    filenameList = [];
    for file in [doc for doc in os.listdir(filepath)
                 if doc.endswith(fileExtension)]:
        filenameList.append(file)
    return filenameList

## Function for when using HTK to get features coefficients
def code_data_to_MFCC(filepath, outputFilePath, configPath, filenameList = None):
    filepath = filepath + "/"
    outputFilePath = outputFilePath + "/"
    if filenameList == None:
        filenameList = find_all_files(filepath, ".wav");
    f = open(filepath + "codetr.scp","w+")
    allOutputFiles = []
    for filename in sorted(filenameList):
        filename = filepath + re.search("(.+).wav",filename).group(1)
        inputFileName = filename + ".wav";
        outputFileName = filename + ".mfc";
        allOutputFiles.append(outputFileName)
        f.write(inputFileName + " " + outputFileName + "\n")
    f.close()
    HTK.HCopy(configPath, filepath+ "codetr.scp")
    htk_reader = HTKFile()
    for filename in sorted(allOutputFiles):
        htk_reader.load(filename)
        result = numpy.array(htk_reader.data)
        result = result[:,:-1]
        filename_out = (outputFilePath  
                       + re.search(".+\/(.+).mfc",filename).group(1)
                       + ".out")    
        numpy.savetxt(filename_out,result,delimiter=",")

