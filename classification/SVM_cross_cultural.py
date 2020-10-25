from matplotlib.pyplot import xlabel, ylabel
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import svm


def create_svm(X_train, X_valid, y_train, y_valid):
	clf = svm.SVC(kernel ='linear',gamma=1.5, C=1.8) #linear kernel seems to work the best for North American Datatset, with gamma = 2 and C = 1.8; Philipines was gamma = 1.5
	clf = clf.fit(X_train, y_train)
	print(clf.score(X_train, y_train))
	predictions = clf.predict(X_valid)
	print(classification_report(y_valid, predictions))
	print(accuracy_score(y_valid, predictions))
	score = accuracy_score(y_valid, predictions)
	fscore = f1_score(y_valid, predictions, average=None)
	return clf, score, fscore



def main():
	
	columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','culture','emotion']
	df = pd.read_csv("../videos_relabelled.csv") 

	#extracting total set for training and testing sets 
	# training and testing on only NA and Persian culture
	df = df[(df['culture']  == 'North America') | (df['culture']  == 'Philippines')] 
	# training and testing on only NA and Philippines culture
	#df = df[(df['culture']  == 'North America') | (df['culture']  == 'Philippines')] 

	#df = df[df['culture'] == 'Persian'] #training and testing on only persian culture
	#df = df[df['culture'] == 'Philippines'] #training and testing on only philipines culture
	#df = df[df['culture'] == 'North America'] #training and testing on only NA culture

	#df['culture_code'] = df['culture'].astype('category').cat.codes

	############# testing model by selecting specific videos to test so components of video are not in training set ###############
	validation_array = []
	test_array = []
	vf_score = []
	tf_score = []
	kfold = KFold(5, True, 1)
	videos = df['filename'].unique()
	print(videos)
	##  Roya: Add label encoder to convert labels to integer
	le = LabelEncoder()
	#this part is for extracting what to test on
	#this is testing set for NA culture
	#test_df = df[(df['filename'] == 'contempt_38') | (df['filename'] == 'contempt_39') | (df['filename'] == 'anger_26') | (df['filename'] == 'anger_27') | (df['filename'] == 'disgust_20') | (df['filename'] == 'disgust_21')  ]
	
	#this is testing set for Persian culture
	# test_df = df[(df['filename'] == '40') | (df['filename'] == '42') | (df['filename'] == '77') | (df['filename'] == '36') | (df['filename'] == '38') | (df['filename'] == '41')  ]
	
	#this is testing set for Filipino culture
	#test_df = df[(df['filename'] == 'contempt_25_p') | (df['filename'] == 'contempt_18_p') | (df['filename'] == 'anger_17_p') | (df['filename'] == 'anger_6_p') | (df['filename'] == 'disgust_7_p') | (df['filename'] == 'disgust_8_p')  ]
	
	#this is for testing on all of Filipino culture
	#test_df = df[df['culture'] == 'Philippines']

	#this is for testing on all of Persian culture
	## Roya: add a test dataframe for displaying results
	test_df = df[df['culture'] == 'Philippines']


	test_videos = test_df['filename'].unique()
	df = df[~df['filename'].isin(list(test_videos))]
	videos = np.array(list(set(videos) - set(test_videos)))
	splits = kfold.split(videos)
	test_df_copy = test_df.drop(['frame', 'face_id', 'culture', 'filename', 'emotion', 'confidence','success'], axis=1)
	for (i, (train, test)) in enumerate(splits):
		print('%d-th split: train: %d, test: %d' % (i+1, len(videos[train]), len(videos[test])))
		train_df = df[df['filename'].isin(videos[train])]
		# test_df = df[df['filename'].isin(videos[test])]
		y = train_df['emotion'].values
		X = train_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
		## Change labels to int using a label encoder
		Y = le.fit_transform(y)

		X_train, X_valid, y_train, y_valid = train_test_split(X, Y)

		#print(X_train)
		print('LABEL ENCODER CLASSES: ', le.classes_)
		clf, score, fscore = create_svm(X_train, X_valid, y_train, y_valid)
		validation_array.append(score)
		vf_score.append(fscore)
		#cv_scores = cross_validate(clf, X, y, cv = 10)
		#print(cv_scores)
		# print(test_df[['frame','filename','culture','emotion']].head())

		int_test = test_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
		# print(len(int_test))
		## Roya: change string labels to integer values
		int_predict = le.fit_transform(test_df['emotion'].values) 
		# print(len(int_predict))
		# predictions = clf.predict(int_test)
		predictions = clf.predict(int_test) #integers predicted
		## Roya: change integer labels to string values
		test_df['predicted'] = le.inverse_transform(predictions)
		## Roya: calculate confusion matrix
		cf_matrix = confusion_matrix(test_df['emotion'].values, test_df['predicted'].values)
		print('CONFUSION MATRIX:\n', cf_matrix)
		df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index=le.inverse_transform([0,1,2]), columns=le.inverse_transform([0,1,2]))
		## Plot Confusion matrix
		plt.figure(figsize=(9,6))
		sn.heatmap(df_cm, annot=True,  fmt='.2%')
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()
		# print("predictions: ", predictions[0:10])
		# print("int_predict: ", int_predict[0:10])
		print(accuracy_score(int_predict, predictions))
		fscore = f1_score(le.fit_transform(int_predict), predictions, average = 'macro')
		test_df.drop(columns=['predicted'], inplace=True)
		print('\n')

		test_array.append(accuracy_score(int_predict, predictions))
		tf_score.append(fscore)


	print("Average accuracy for all Folds on valid dataset: " + str(np.mean(validation_array)))

	print("Average accuracy for all Folds on test dataset: " + str(np.mean(test_array)))

	print("Average f-score for all Folds on valid dataset: " + str(np.mean(vf_score)))

	print("Average f-score for all Folds on test dataset: " + str(np.mean(tf_score)))
	
if __name__=='__main__':
	#train_data = sys.argv[1]
	#test_data = sys.argv[2]
	main()