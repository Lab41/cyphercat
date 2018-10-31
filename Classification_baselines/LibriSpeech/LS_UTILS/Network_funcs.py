import os
import numpy as np
import soundfile as sf            # To read .flac files. 
import librosa

# For the neural network.
# Install PyBrain, e.g. pip install pybrain.
from pybrain.datasets                import ClassificationDataSet
from pybrain.tools.shortcuts         import buildNetwork
from pybrain.supervised.trainers     import BackpropTrainer
from pybrain.structure.modules       import SoftmaxLayer
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter

from pybrain.structure import FullConnection
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import TanhLayer

# Read data from a folder into a list.
def getData(division,speaker,datapath,audioType,durationCheck,deltaT,lim1,lim2,numFeatures,noisy,transform):
    #deltaT is the duration of the audio frame. Lim1 & Lim2 are the frequency ranges; each frequency is a feature. 
    #Noisy sets the limit for pauses in speech 
    #Division is the data, i.e. Train, CV, test
  fname = datapath+division+speaker  
  subPaths = [v+"/" for v in os.listdir(fname) if v[0] != "."]
  dataFiles = []
  for subPath in subPaths:
    files = [v for v in os.listdir(fname+subPath) if v[0] != "." and audioType in v]
    for fil in files:
      data,samplerate = sf.read(fname+subPath+fil)  
      duration = len(data)*1./samplerate
      if duration >= durationCheck: dataFiles.append(fname+subPath+fil)
        
  chunksF = []

  for fil in dataFiles:
    data,samplerate = sf.read(fil)  
    duration = len(data)*1./samplerate

    # Divide audio data into frames, or chunks. 
    numChunks = int(duration/deltaT)
    sizeChunk = int(len(data)/numChunks)
    for lp in range(0,numChunks):    
      chunk = data[lp*sizeChunk:(lp+1)*sizeChunk]      # get a chunk of speech.  
      # np.fft.rfft computes the one-dimensional discrete Fourier Transform of the data
      if transform == 'Fourier':
          chunksF.append(np.abs(np.fft.rfft(chunk))[lim1:lim2])  # take the FFT.
      elif transform == 'Mel':
          S = librosa.feature.melspectrogram(y=chunk, sr=samplerate, n_mels=128, fmax=lim2)
          chunksF.append(np.abs(S))

    # Delete quiet parts of speech, i.e. pauses.
    # Most of the power is in the bottom 50% of frequencies.
    mu = np.mean([np.mean(chunksF[i][:numFeatures//2]) for i in range(0,len(chunksF))])
    dataF = []
    for chunkF in chunksF:
      if np.mean(chunkF[:numFeatures//2]) > noisy*mu:
        dataF.append(chunkF)
    
  return dataF

# Return data for all speakers.
def getDataSpeakers(division,speakers,datapath,audioType,durationCheck,deltaT,lim1,lim2,numFeatures,noisy,transform):
  dataSpeakers = []
  for speaker in speakers:
    #print("Getting data for speaker: "+speaker)
    dataSpeakers.append(getData(division,speaker,datapath,audioType,durationCheck,deltaT,lim1,lim2,numFeatures,noisy, transform))

  N = np.sum([np.shape(s)[0] for s in dataSpeakers])
  tX = np.mat(np.zeros((N,numFeatures)))
  tY = []
  speakerIndices = [0]    # Index corresponding to start of speaker 'n'
  
  ctr = 0; lp = 0
  for dataSpeaker in dataSpeakers:
    for j in range(0,len(dataSpeaker)):
      for k in range(0,numFeatures):
        tX[ctr,k] = dataSpeaker[j][k]
      tY.append(lp)
      ctr += 1  
    speakerIndices.append(ctr)
    lp += 1  
          
  return tX,tY,speakerIndices

# This is the architecture of the network.
# Hyper-parameters to be fixed through cross-validation are:
#                         (i)   How many layers are necessary?
#                         (ii)  How many nodes per layer?
#                         (iii) What kind of activation function to use?
# 
def setupNetwork(numHiddenNodes,numHiddenLayers,numFeatures,numSpeakers):
    
  nn = FeedForwardNetwork()
  inputLayer = LinearLayer(numFeatures)
  nn.addInputModule(inputLayer)
  
  hiddenLayers = []
  for x in range(numHiddenLayers):
    hiddenLayer = TanhLayer(numHiddenNodes)    
    nn.addModule(hiddenLayer)
    hiddenLayers.append(hiddenLayer)
  outputLayer = SoftmaxLayer(numSpeakers)
  nn.addOutputModule(outputLayer)
  
  inputConnection = FullConnection(inputLayer,hiddenLayers[0])
  nn.addConnection(inputConnection)
  
  for x in range(numHiddenLayers-1):
    connect = FullConnection(hiddenLayers[x],hiddenLayers[x-1])
    nn.addConnection(connect)

  outputConnection = FullConnection(hiddenLayers[numHiddenLayers-1],outputLayer)    
  nn.addConnection(outputConnection)
  nn.sortModules()
  
  return nn


# Test the classifier.
# nns is a list of trained networks. It is sometimes helpful to pass more than one network,
# since errors made by different networks may cancel out.
# tX: Data to test
# tY: Target, i.e. speaker ID.
# idx: List of indices indicating the starting location of a speaker.
# skip: Number of increments of 'deltaT' to group together. For e.g. if 'deltaT' = 0.2:
#       If skip = 1, a prediction is made for every 0.2 seconds.
#       If skip = 5, a prediction is made for every 1.0 second.
#
def tstClassifier(nns,tX,tY,idx,skip,numSpeakers,numFeatures):

  def maxIdx(A):
    # Pick the prediction with the highest occurance.
    ctr = {}
    for pre in A:
      if pre not in ctr: ctr[pre] = 1
      else: ctr[pre] += 1
      
    rev = {}
    for key in ctr.keys():
      rev[ctr[key]] = key
    return rev[np.max(list(rev.keys()))]

  # Confusion matrix: Speaker 'm' predicted as speaker 'n'.
  confusion = np.mat(np.zeros((numSpeakers,numSpeakers)))  
  
  correct = 0; al = 0
  for cvi in range(0,numSpeakers):
    # idx contain the start location of each speaker.
    for lpx in range(idx[cvi],idx[cvi+1]-skip,skip):
      bestArray = []

      # Consider "skip" number of data points together.
      for lp in range(lpx,lpx+skip):      
        A = [tX[lp,i] for i in range(0,numFeatures)]
        prediction = []

        # Average over multiple trained networks.        
        for nn in nns:  
          pred = nn.activate(A)
          ctr = {}
          for i in range(0,numSpeakers):
            ctr[pred[i]] = i
          prediction.append(ctr[np.max(list(ctr.keys()))])
        bestArray.append(maxIdx(prediction))

      best = maxIdx(bestArray)                
      if best == tY[lpx]: correct += 1

      # Populate the confusion matrix.  
      for i in range(0,numSpeakers):
        if best == i: confusion[cvi,i] += 1
      al += 1

  return correct*1./al, confusion