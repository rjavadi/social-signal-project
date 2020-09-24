from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm


def create_svm(X_train, X_valid, y_train, y_valid):
	#scaler = StandardScaler().fit(X_train)
	#X_scaled = scaler.transform(X_train)
	#X_valid_scaled = scaler.transform(X_valid)
	clf = svm.SVC(kernel ='rbf',gamma=1.5, C=1.8)
	#clf = svm.SVC(kernel='linear')

	#clf = AdaBoostClassifier(clf, learning_rate=0.6,  algorithm='SAMME')
	#cv_scores = cross_validate(clf, X, y, cv = 10, scoring = ['recall', 'precision','accuracy'])
	

	clf = clf.fit(X_train, y_train)
	print(clf.score(X_train, y_train))
	#print(clf.score(X_valid, y_valid))
	predictions = clf.predict(X_valid)
	print(classification_report(y_valid, predictions))
	print(accuracy_score(y_valid, predictions))
	score = accuracy_score(y_valid, predictions)
	fscore = f1_score(y_valid, predictions, average=None)
	return clf, score, fscore


def separate_emotions(X):


	print( '############# PHILIPPINES############### \n')
	#################### PHILIPPINES ###################
	culture_1 = X[(X['culture'] == 'Persian') | (X['culture'] == 'North America')]
	test = X[X['culture'] == 'Philippines']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print('\n')


	print( '############# NORTH AMERICA ############### \n')
	#################### NORTH AMERICA ###################
	culture_1 = X[(X['culture'] == 'Persian') | (X['culture'] == 'Philippines')]
	test = X[X['culture'] == 'North America']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print('\n')
	print( '############# PERSIAN ############### \n')
	#################### NORTH AMERICA ###################
	culture_1 = X[(X['culture'] == 'North America') | (X['culture'] == 'Philippines')]
	test = X[X['culture'] == 'Persian']
	culture_1['culture_code'] = culture_1['culture'].astype('category').cat.codes
	y = culture_1['emotion'].values
	culture_1 = culture_1.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	
	X_train, X_valid, y_train, y_valid = train_test_split(culture_1, y)
	clf = create_svm(X_train, X_valid, y_train, y_valid)

	test['culture_code'] = test['culture'].astype('category').cat.codes
	int_test = test.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	print(len(int_test))
	int_predict = test['emotion'].values
	print(len(int_predict))
	predictions = clf.predict(int_test)
	print(accuracy_score(int_predict, predictions))
	print(classification_report(int_predict, predictions))
	print('\n')




def main():
	
	columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r','culture','emotion']
	df = pd.read_csv("all_videos.csv")
	#df['culture_code'] = df['culture'].astype('category').cat.codes


	############# testing model by separating training on 2 cultures and test on 1 culture####################
	#separate_emotions(df)
	

	############# testing model by selecting specific videos to test so components of video are not in training set ###############
	validation_array = []
	test_array = []
	vf_score = []
	tf_score = []
	kfold = KFold(5, True, 1)
	videos = df['filename'].unique()
	# test_videos = pd.Series(videos).sample(frac=0.10)
	# print(test_videos)
	# videos must be array to be subscriptable by a list
	# Removing test videos from train dataset
	# test_df = df[df['filename'].isin(test_videos)]
	test_df = df[df['culture'] == 'North America']
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

	    X_train, X_valid, y_train, y_valid = train_test_split(X, y)


	    clf, score, fscore = create_svm(X_train, X_valid, y_train, y_valid)
	    validation_array.append(score)
	    vf_score.append(fscore)
	    #cv_scores = cross_validate(clf, X, y, cv = 10)
	    #print(cv_scores)
	    print(test_df[['frame','filename','culture','emotion']].head())

	    int_test = test_df.drop(columns = ['success','confidence', 'face_id','frame','emotion', 'culture','filename']).values
	    print(len(int_test))
	    int_predict = test_df['emotion'].values
	    print(len(int_predict))
	    predictions = clf.predict(int_test)
	    print(predictions[0:10])
	    print(accuracy_score(int_predict, predictions))
	    fscore = f1_score(int_predict, predictions, average = 'macro')

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


