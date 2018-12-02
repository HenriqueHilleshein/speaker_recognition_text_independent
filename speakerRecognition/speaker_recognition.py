# -*- coding: utf-8 -*-
import numpy
import myUtils as ut
from collections import defaultdict
from sklearn.mixture import GaussianMixture

class SpeakerRecognition:
    def __init__(self, speakerIdRegex = "(.+)_"):
        self.__speakerIdRegex = speakerIdRegex
        self.__UBMModel = None
        self.__allModels = defaultdict(GaussianMixture)

    def get_gmm(self):
        return GaussianMixture(n_components=12, covariance_type="full", 
                               tol=0.001, reg_covar=1e-06, max_iter=100, 
                               n_init=1, init_params="kmeans", 
                               weights_init=None, means_init=None, 
                               precisions_init=None, random_state=None, 
                               warm_start=False, verbose=0, 
                               verbose_interval=10)

    def remove(self, speakerId):
        if speakerId in self.get_all_trained_speakers_id():
            del self.__allModels[speakerId]
            
    def train(self, trainingPath, filenameList = None, speakersId = None):
        trainCoefficientsDict = ut.load_coefficients(trainingPath,
                                                     filenameList=filenameList,
                                                     speakersId=speakersId)

        speakersId = list(trainCoefficientsDict.keys())
        if speakersId == []:
            return
        for speaker in speakersId:
            melCoefficientsNum = trainCoefficientsDict[speaker][0][0].size;
            allTextSpeech = numpy.empty((0,melCoefficientsNum));
            for textSpeech in trainCoefficientsDict[speaker]:
                if allTextSpeech.any():
                    allTextSpeech = numpy.concatenate((allTextSpeech
                                                       ,textSpeech), axis=0)
                else:
                    allTextSpeech = textSpeech
            gmm = self.get_gmm()
            gmm.fit(allTextSpeech)
            self.__allModels[speaker] = gmm
    
    def test(self, speechCoefficients, speakerId):
        modelScore = self.__allModels[speakerId].score(speechCoefficients)
        backgroundScore = None
        likelihoodDifference = None
        if(self.__UBMModel is not None):
            backgroundScore = self.__UBMModel.score(speechCoefficients)
            likelihoodDifference = modelScore - backgroundScore
        return modelScore, backgroundScore, likelihoodDifference
        
    def create_ubm_model(self, trainingPath, filenameList = None,
                         speakersId = None):
        trainCoefficientsDict = ut.load_coefficients(trainingPath,
                                                     filenameList=filenameList,
                                                     speakersId=speakersId)
        speakersId = list(trainCoefficientsDict.keys())
        if speakersId == []:
            return
        melCoefficientsNum = trainCoefficientsDict[speakersId[0]][0][0].size;
        allTextSpeech = numpy.empty((0,melCoefficientsNum));
        for speaker in speakersId:
            for textSpeech in trainCoefficientsDict[speaker]:
                if allTextSpeech.any():
                    allTextSpeech = numpy.concatenate((allTextSpeech
                                                       ,textSpeech), axis=0)
                else:
                    allTextSpeech = textSpeech
        self.__UBMModel = self.get_gmm()
        self.__UBMModel.fit(allTextSpeech)    

    def get_all_trained_speakers_id(self):
        speakersId = list(self.__allModels.keys())
        return speakersId
    
    def verification(self, speechCoefficients, speakerId, deltaV = 0):
        testResult = self.test(speechCoefficients, speakerId)
        if testResult[2] == None:
            return False
        if(testResult[2] > deltaV):
            return True
        return False
    
    def identification(self, speechCoefficients, isOpenSet):
        Result = [-200,""]
        for speaker in self.get_all_trained_speakers_id():
            testValue = self.test(speechCoefficients, speaker)
            if Result[0] < testValue[0]:
                Result = [testValue[0], speaker]
        if isOpenSet is True:
            if self.verification(speechCoefficients, Result[1]) is False:
                return ""
        return Result[1]
    
    def find_speaker(self, speechCoefficients, speakerId, samplesNumber = 100,
                     slideMode = False):
        speechSamplesNum = speechCoefficients[:,1].size
        if samplesNumber > speechSamplesNum:
            return -1
        endAdjustment = speechSamplesNum - speechSamplesNum%samplesNumber
        iterator = 0
        while (iterator + samplesNumber) < speechSamplesNum:
            lastSample = iterator+samplesNumber-1
            coefficients4Test = speechCoefficients[iterator:lastSample,:] 
            if self.identification(coefficients4Test, True) == speakerId:
                return iterator
            if slideMode is True:
                iterator = iterator + 1
            else:
                iterator = iterator + samplesNumber
                if iterator == endAdjustment:
                    adjustment = samplesNumber - speechSamplesNum%samplesNumber
                    iterator = iterator - adjustment
        return -1
        
    
    def detection(self, speechCoefficients, speakerId, timeWindow = 1,
                  featureWindow = 0.020, slideMode = False):
        samplesNumber = int(timeWindow/featureWindow)
        pos = self.find_speaker(speechCoefficients, speakerId, samplesNumber,
                                slideMode)
        if pos == -1:
            return False
        return True
    
    def tracking(self, speechCoefficients, speakerId, timeWindow = 1,
                 featureWindow = 0.020, slideMode = False):
        segments = self.segmentation(speechCoefficients, timeWindow
                                    , featureWindow, slideMode)
        segmentsNum = len(segments)
        trackReturn = list()
        for i in range(segmentsNum):
            if segments[i][0] == speakerId:
                if i < (segmentsNum - 1):
                    trackReturn.append([segments[i][1],segments[i+1][1]])
                else:
                    speechSamplesNum = speechCoefficients[:,1].size
                    trackReturn.append([segments[i][1], 
                                        (speechSamplesNum)*featureWindow])
        return trackReturn

    
    def segmentation(self, speechCoefficients, timeWindow = 1,
                     featureWindow = 0.020, slideMode = False):    
        samplesNumber = int(timeWindow/featureWindow)    
        speechSamplesNum = speechCoefficients[:,1].size
        speakerList = list()
        if samplesNumber > speechSamplesNum:
            speakerList.append(["",0])
            return speakerList
        endAdjustment = speechSamplesNum - speechSamplesNum%samplesNumber
        iterator = 0
        lastSpeaker = ""
        while (iterator + samplesNumber) < speechSamplesNum:
            lastSample = iterator+samplesNumber
            coefficients4Test = speechCoefficients[iterator:lastSample,:]
            speaker = self.identification(coefficients4Test, True)
            if speaker != lastSpeaker:
                speakerList.append([speaker,iterator*featureWindow])
                lastSpeaker = speaker
            if slideMode is True:
                iterator = iterator + 1
            else:
                iterator = iterator + samplesNumber
                if iterator == endAdjustment:
                    adjustment = samplesNumber - speechSamplesNum%samplesNumber
                    iterator = iterator - adjustment
        if speakerList == []:
            speakerList.append(["",0])
            return speakerList
        if(speakerList[0][1] != 0):
            speakerList.insert(0,["",0])
        return speakerList