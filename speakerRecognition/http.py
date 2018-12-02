# -*- coding: utf-8 -*-
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
import wave
import audioop
import re


WEB_PATH = "/opt/speakerRecognition/web/" # Where is the web page files
TRAIN_TEXTS_PATH = WEB_PATH + "train_texts/"  #Where is the text for the training

from speaker_recognition import SpeakerRecognition
import myUtils as ut

TRAINING_PATH = "../../ELSDSR/train/" # Where is stored the training audio
TEST_PATH = "../../ELSDSR/test/" # Where is stored the test audio
CONFIG_PATH = "../../ELSDSR/config" # HTK file location
COEFFICIENTS_TRAINING_PATH = TRAINING_PATH + "coefficients2/" # Where is stored the training coefficients(MFCC)
COEFFICIENTS_TEST_PATH = TEST_PATH + "coefficients2/"  # Where is stored the test coefficients(MFCC)

sv = SpeakerRecognition()

def train_new_id(speaker):
    allFiles = ut.find_all_files(TRAINING_PATH, ".wav")
    for file in allFiles:
        match = re.search("trainwav_(.+)\.wav", file)
        if match is None:
            continue
        downsampleWav(TRAINING_PATH + file, TRAINING_PATH + speaker + "_" 
                      + match.group(1) + ".wav")
        ut.code_data_to_MFCC(TRAINING_PATH, COEFFICIENTS_TRAINING_PATH, 
                             CONFIG_PATH, [speaker + "_" + match.group(1)
                                           + ".wav"])    
    sv.train(trainingPath=COEFFICIENTS_TRAINING_PATH, speakersId=[speaker])
    
def prepare_for_test(typeTest):
    downsampleWav(TRAINING_PATH + "mywav.wav", TEST_PATH + typeTest + ".wav")
    ut.code_data_to_MFCC(TEST_PATH, COEFFICIENTS_TEST_PATH, CONFIG_PATH,
                         [typeTest + ".wav"])
    coefficients = ut.load_one_file_cofficients(COEFFICIENTS_TEST_PATH + 
                                                typeTest + ".out")
    return coefficients

def get_all_training_texts():
    trainTextsList = ut.find_all_files(TRAIN_TEXTS_PATH, ".txt")
    data = {}
    for trainText in trainTextsList:
        with open(TRAIN_TEXTS_PATH + trainText, 'r') as myfile:
            data[trainText] = myfile.read()
            myfile.close()
    return json.dumps(data)

def get_all_available_IDs():
    allFiles = ut.find_all_files(COEFFICIENTS_TRAINING_PATH, ".out")
    trainedSpeakersId = sv.get_all_trained_speakers_id()
    data = {}
    for file in allFiles:
        match = re.search("(.+)_.+\.out", file)
        if match is None:
            continue
        speakerId = match.group(1)
        if speakerId in trainedSpeakersId:
            data[speakerId] = True
        else:
            data[speakerId] = False
    return json.dumps(data)

def train_by_speaker_id_list(speakerIdList):
    speakerIdListJson = json.loads(speakerIdList)
    trainedSpeakersId = sv.get_all_trained_speakers_id()
    speakerId2beTrained = list()

    for speakerId in speakerIdListJson.keys():
        if speakerIdListJson[speakerId] == True:
            if speakerId not in trainedSpeakersId:
                speakerId2beTrained.append(speakerId)
        else:
            if speakerId in trainedSpeakersId:
                sv.remove(speakerId)
    
    sv.train(trainingPath=COEFFICIENTS_TRAINING_PATH
             , speakersId=speakerId2beTrained)
    
def train_ubm_model(speakerIdList):
    speakerIdListJson = json.loads(speakerIdList)
    speakerId2beTrained = list()
    for speakerId in speakerIdListJson.keys():
        if speakerIdListJson[speakerId] == True:
            speakerId2beTrained.append(speakerId)
    sv.create_ubm_model(trainingPath=COEFFICIENTS_TRAINING_PATH
                        , speakersId=speakerId2beTrained)

def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=1, outchannels=1):
    if not os.path.exists(src):
        print('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1 and inchannels > 2:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print('Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted[0])
    except:
        print('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print('Failed to close wav files')
        return False

    return True




class S(BaseHTTPRequestHandler):
    def _set_headers(self, code):
        self.send_response(code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):

        path = self.path.split("?")[0]
        path = path[1:]
        if path == "":
            path = "index.html"
        if path == "trainingTexts.json":
            self._set_headers(200)
            message = get_all_training_texts()
        elif path == "allavailableIDs.json":
            self._set_headers(200)
            message = get_all_available_IDs()
        else:
            try:
                with open(WEB_PATH + path, 'r') as myfile:
                    self._set_headers(200)
                    message=myfile.read()
                    myfile.close()
            except IOError:
                self._set_headers(404)
                message = "NOT FOUND"
        self.wfile.write(bytes(message, "utf8"))
        
    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers(200)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        message = ""
        path = self.path[1:]
        if path.endswith(".wav"):
            file = open(TRAINING_PATH + path, "wb")
            file.write(post_data)
            file.close()
        elif path == ("action.json"):
            actionJson = json.loads(post_data)
            if actionJson['action'] == "train":
                speaker = actionJson["id"]
                train_new_id(speaker)
                message = ""
            if actionJson['action'] == "verification":
                speaker = actionJson["id"]
                coefficients = prepare_for_test("verification")
                if sv.verification(coefficients, speaker):
                    message = "Verified"
                else:
                    message = "Impostor"
            if actionJson['action'] == "closedset":
                coefficients = prepare_for_test("closedset")
                speaker = sv.identification(coefficients, False)
                message = speaker
            if actionJson['action'] == "openset":
                coefficients = prepare_for_test("openset")
                speaker = sv.identification(coefficients, True)
                if speaker == "":
                    speaker = "Speaker not found"
                message = speaker
            if actionJson['action'] == "detection":
                speaker = actionJson["id"]
                coefficients = prepare_for_test("detection")
                if(sv.detection(coefficients, speaker, timeWindow=1
                                , featureWindow=0.02, slideMode=False)):
                    message = "Detected"
                else:
                    message = "Not detected"
            if actionJson['action'] == "tracking":
                speaker = actionJson["id"]
                coefficients = prepare_for_test("tracking")
                result = sv.tracking(coefficients, speaker, timeWindow=1
                                     , featureWindow=0.02, slideMode=False)
                if result:
                    for period in result:
                        message += "<p>"
                        message += str(period[0]) + "-" + str(period[1])
                        message += "</p>"
            if actionJson['action'] == "segmentation":
                coefficients = prepare_for_test("segmentation")
                result = sv.segmentation(coefficients, timeWindow=1
                                         , featureWindow=0.02, slideMode=False)
                if result:
                    for speakerTime in result:
                        if speakerTime[0] == "":
                            speakerTime[0] = "Unkown"
                        message += "<p>"
                        message += speakerTime[0] + "-" + str(speakerTime[1])
                        message += "</p>"

        elif path == "trainlist.json":
            train_by_speaker_id_list(post_data)
        elif path == "trainubmlist.json":
            train_ubm_model(post_data)
        self.wfile.write(bytes(message, "utf8"))
        
def run(server_class=HTTPServer, handler_class=S, port=32000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
