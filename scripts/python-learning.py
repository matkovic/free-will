import numpy as np
import pandas
from math import ceil, floor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold, permutation_test_score
from sklearn import linear_model
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import numpy.fft as fft
from sklearn import datasets

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		# Names of CSV files
		self.signalFileName = ""
		self.stimFileName = ""
		# Name of Save File
		self.saveFileName = ""
		# other variables
		self.windowSize = 0  # window size in ms
		self.numOfPreviousWindowsAsOne = 0  # number of windows before actual stimulation to be marked as 1
		self.numOfWindowsBefore = 0  # number of windows before those marked as 1, to be marked as 0
		self.numOfWindowsAfter = 0  # number of windows after those marked as 1, to be marked as 0
		
		
	def filter_signal(self, sampleRate, numberOfSamplesWindow, stimulationTimes, splittedSignal):
		""" returns tuple of filtered (signal chunks, classes) """
		splittedSignal_filtrd = []
		classes_filtrd = []
		
		temp_classes = np.zeros(len(splittedSignal))
		
		for stim in stimulationTimes:
			index = int(floor(stim*sampleRate/numberOfSamplesWindow))
			temp_classes[index] = 1
			for i in range(1, self.numOfPreviousWindowsAsOne):
				temp_classes[index-i] = 1
			
			tmp_cls_winds = temp_classes[(index - self.numOfPreviousWindowsAsOne - self.numOfWindowsBefore):index+self.numOfWindowsAfter]
			tmp_sig_winds = np.concatenate(splittedSignal[(index - self.numOfPreviousWindowsAsOne - self.numOfWindowsBefore):index+self.numOfWindowsAfter])
			
			if len(tmp_sig_winds)/len(tmp_cls_winds)!=numberOfSamplesWindow:  # if np.array_split does not split in equal windows
				tmp_sig_winds = np.lib.pad(tmp_sig_winds, ((0, int(len(tmp_cls_winds)*numberOfSamplesWindow-len(tmp_sig_winds))),(0, 0)), 'edge')  # pad with same values on end of array
			
			classes_filtrd.extend(tmp_cls_winds)
			splittedSignal_filtrd.extend(np.array_split(tmp_sig_winds, len(tmp_cls_winds)))		
		
		return (splittedSignal_filtrd, classes_filtrd)

		
	def avg_k_fold(self, data, classes, k=4):
		""" return average CA of k-fold cross validation """
		avg_val = 0
	
		kf = KFold(n_splits=k)
		for train, test in kf.split(data):
			clf = svm.SVC(kernel='linear', C=1).fit(data[train], classes[train])
			cur_score = clf.score(data[test], classes[test])
			avg_val += cur_score
			# print cur_score
		return avg_val/k

		
	def permutation_significance_classification_score(self, X, y, k_folds=4):
		n_classes = np.unique(y).size
		
		svm = SVC(kernel='linear')
		cv = StratifiedKFold(k_folds)

		score, permutation_scores, pvalue = permutation_test_score(svm, X, y, scoring="accuracy", cv=cv, n_permutations=200, n_jobs=1)
		print("Classification score %s (pvalue : %s)" % (score, pvalue))
		
		plt.hist(permutation_scores, 20, label='Permutation scores')
		ylim = plt.ylim()
		plt.plot(2 * [score], ylim, '--g', linewidth=3,
				 label='Classification Score'
				 ' (pvalue %s)' % pvalue)
		plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Luck')

		plt.ylim(ylim)
		plt.legend()
		plt.xlabel('Score')
		plt.show()
		
			
	def initialize(self):
		# Names of CSV files
		self.signalFileName = self.setting['InputCSVSignal']
		self.stimFileName = self.setting['InputCSVStimulations']
		# Name of Save File
		self.saveFileName = self.setting['SaveFile']
		# other variables
		self.windowSize = int(self.setting['WindowSize (ms)'])  # in ms
		self.numOfPreviousWindowsAsOne = int(self.setting['NumOfPrevWindows'])
		self.numOfWindowsBefore = int(self.setting['NumOfWindowsBefore'])-1
		self.numOfWindowsAfter = int(self.setting['NumOfWindowsAfter'])+1
		self.k_folds = int(self.setting['K-folds'])
		
		print "Reading files..."
		# read CSV files
		signalArray = pandas.read_csv(self.signalFileName, delimiter=";", encoding="utf-8-sig")
		stimsArray = pandas.read_csv(self.stimFileName, delimiter=";", encoding="utf-8-sig")
		print "Files read!"
		
		# sort information from tables (pandas dataframe)
		time = signalArray['Time (s)']
		electrodes = signalArray.iloc[:, 1:signalArray.shape[1]-1]
		sampleRate = signalArray['Sampling Rate'][0]
		stimulationTimes = stimsArray['Time (s)']
		
		numberOfSamplesWindow = floor(self.windowSize / ((1/sampleRate)*1000))  # number of samples for approximately self.windowSize ms
		
		splittedSignal = np.array_split(electrodes, ceil(len(time)/numberOfSamplesWindow))  # split signal into chunks of specified length
		
		s_c = self.filter_signal(sampleRate, numberOfSamplesWindow, stimulationTimes, splittedSignal)  # filter signal and return sparsed version of chunks and assign classes
		splittedSignal_filtrd = s_c[0]
		classes_filtrd = s_c[1]

		splittedSignal_filtrd_means = np.array(np.mean(splittedSignal_filtrd, axis=1))  # for each window calculate mean value
		# additional attributes could be added besides splittedSignal_filtrd_means
		classes_filtrd = np.array(classes_filtrd)
		
		# print average k-fold CA
		print "Average " + str(self.k_folds) + "-folds value: " + str(self.avg_k_fold(splittedSignal_filtrd_means, classes_filtrd, k=self.k_folds))
		#self.permutation_significance_classification_score(splittedSignal_filtrd_means, classes_filtrd, k_folds=2)  # k_folds=2, last long if k is bigger
		
		if self.saveFileName:
			clf = svm.SVC(kernel='linear', C=1).fit(splittedSignal_filtrd_means, classes_filtrd)
			clf.fit(splittedSignal_filtrd_means, classes_filtrd)
			
			print "Saving to pickle file: " + self.saveFileName
			pickle.dump(clf, open(self.saveFileName, 'wb'))
			print "Learned score: " + str(clf.score(splittedSignal_filtrd_means, classes_filtrd))
		
		# send finish stimulation output (for OpenViBE)
		self.finishBySendingStimulation(32774)  # OVTK_StimulationId_TrialStop code
		
		
	def finishBySendingStimulation(self, stimulationCode):
		stimSetFinish = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
		stimSetFinish.append(OVStimulation(stimulationCode, self.getCurrentTime(), 0.))
		self.output[0].append(stimSetFinish)

		
	def process(self):
		
		return
		
	def uninitialize(self):
		# nop
		
		return

box = MyOVBox()
