import sys
import pandas as pd
import numpy as np
import gzip
import read_write as rw
import LDA as lda

'''
finput_title = "Data/title"
finput_description = "Data/description"
finput_train_item_id = "Data/train_item_id"
finput_test_item_id = "Data/test_item_id"
foutput_title_similarity_train = "Data/title_similarity_matrix_train"
foutput_title_similarity_test = "Data/title_similarity_matrix_test"
foutput_description_similarity_train = "Data/description_similarity_matrix_train"
foutput_description_similarity_test = "Data/description_similarity_matrix_test"
'''

if (__name__ == '__main__'):
	#### data path
	finput_topic_num = int(sys.argv[1])
	finput_title = sys.argv[2]
	finput_description = sys.argv[3]
	finput_train_item_id = sys.argv[4]
	finput_test_item_id = sys.argv[5]
	foutput_title_similarity_train = sys.argv[6]
	foutput_title_similarity_test = sys.argv[7]
	foutput_description_similarity_train = sys.argv[8]
	foutput_description_similarity_test = sys.argv[9]

	#### read into item title and description information (dict: {id : content})
	dict_title = rw.readffile(finput_title)
	dict_description = rw.readffile(finput_description)

	#### preprocess before LDA
	dict_title_preprocessed = lda.texts_preprocess(dict_title)
	dict_description_preprocessed = lda.texts_preprocess(dict_description)
	list_title_preprocessed = list(dict_title_preprocessed.values())
	list_description_preprocessed = list(dict_description_preprocessed.values())
	print("text preprocessed done!")

	#### generate item title and description similarity for all items
	title_similarity = lda.LDA(texts=list_title_preprocessed, num_topics=finput_topic_num)
	description_similarity = lda.LDA(texts=list_description_preprocessed, num_topics=finput_topic_num)
	print("lda similarity calculated done!")

	#### generate train/test item similarity matrix
	df_title_similarity_matrix = pd.DataFrame(np.array(title_similarity),index=list(dict_title.keys()),columns=list(dict_title.keys()))
	df_description_similarity_matrix = pd.DataFrame(np.array(description_similarity),index=list(dict_description.keys()),columns=list(dict_description.keys()))
	train_item_id = rw.readffile(finput_train_item_id)
	test_item_id = rw.readffile(finput_test_item_id)
	#### title/train
	df_title_similarity_matrix_train = df_title_similarity_matrix.loc[list(train_item_id.keys()), list(train_item_id.keys())]
	#### title/test
	df_title_similarity_matrix_test = df_title_similarity_matrix.loc[list(test_item_id.keys()), list(test_item_id.keys())]
	#### description/train
	df_description_similarity_matrix_train = df_description_similarity_matrix.loc[list(train_item_id.keys()), list(train_item_id.keys())]
	#### description/test
	df_description_similarity_matrix_test = df_description_similarity_matrix.loc[list(test_item_id.keys()), list(test_item_id.keys())]
	print("similarity matrix generated done!")

	#### write data into files
	rw.write2file(df_title_similarity_matrix_train, foutput_title_similarity_train)
	rw.write2file(df_title_similarity_matrix_test, foutput_title_similarity_test)
	rw.write2file(df_description_similarity_matrix_train, foutput_description_similarity_train)
	rw.write2file(df_description_similarity_matrix_test, foutput_description_similarity_test)
	print("file saved done!")