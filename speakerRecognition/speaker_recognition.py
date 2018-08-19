# -*- coding: utf-8 -*-
import numpy
import myUtils as ut
from collections import defaultdict
from sklearn.mixture import GaussianMixture

class SpeakerRecognition:
    def __init__(self, speakerIdRegex = "(.+)_", verificationDelta = 0):
        self.__speakerIdRegex = speakerIdRegex
        self.__UBMModel = None
        self.__allModels = defaultdict(GaussianMixture)
        self.__verificationDelta = verificationDelta

    def get_gmm(self):
        return GaussianMixture(n_components=12, covariance_type="full", 
                              tol=0.001, reg_covar=1e-06, max_iter=100, 
                              n_init=1, init_params="kmeans", 
                              weights_init=None, means_init=None, 
                              precisions_init=None, random_state=None, 
                              warm_start=False, verbose=0, 
                              verbose_interval=10)

    def train(self, trainingPath, filenameList = None, speakersId = None, 
                     isUBM = False):
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
    
    def test(self, speakerId, speechCoefficients):
        modelScore = self.__allModels[speakerId].score(speechCoefficients)
        backgroundScore = None
        likelihoodDifference = None
        if(self.__UBMModel is not None):
            backgroundScore = self.__UBMModel.score(speechCoefficients)
            likelihoodDifference = modelScore - backgroundScore
        return modelScore, backgroundScore, likelihoodDifference
        

    def create_ubm_model(self, trainingPath, filenameList = None, speakersId = None):
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
    
    def verification(self, speakerId, speechCoefficients):
        testResult = self.test(speakerId, speechCoefficients)
        if(testResult[2] > self.__verificationDelta):
            return True
        return False
    
    def identification(self, speechCoefficients, isOpenSet):
        Result = [-200,""]
        for speaker in self.get_all_trained_speakers_id():
            testValue = self.test(speaker, speechCoefficients)
            if Result[0] < testValue[0]:
                Result = [testValue[0], speaker]
        if isOpenSet is True:
            if self.verification(Result[1], speechCoefficients) is False:
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
        
    
    def detectation(self, speechCoefficients, speakerId, timeWindow = 1
                    , featureWindow = 0.020, slideMode = False):
        samplesNumber = int(timeWindow/featureWindow)
        pos = self.find_speaker(speechCoefficients, speakerId, samplesNumber,
                                slideMode)
        if pos == -1:
            return False
        return True
    
    def tracking(self, speechCoefficients, speakerId, timeWindow = 1
                , featureWindow = 0.020, slideMode = False):
        samplesNumber = int(timeWindow/featureWindow)
        pos = self.find_speaker(speechCoefficients, speakerId, samplesNumber,
                                slideMode)
        if pos == -1:
            return -1
        return pos*featureWindow
    
    def segmentation(self, speechCoefficients, timeWindow = 1
                    , featureWindow = 0.020, slideMode = False):    
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
        if(speakerList[0][1] != 0):
            speakerList.insert(0,["",0])
        return speakerList