import sys
import pandas as pd
import numpy as np
import read2df as rdf
import read_write as rw

'''
Input: input path ("../Dataset/All_Beauty/meta_All_Beauty.json.gz")
	   output path ("Data/title" && "Data/description")
output: files
'''
if (__name__ == '__main__'):
	#### data path
	finput = sys.argv[1]
	foutput_title = sys.argv[2]
	foutput_description = sys.argv[3]

	#### read data into dataframe
	df = rdf.getDF(finput)

	#### delete rows where title or description is nan
	dict_title = {}
	dict_description = {}
	subdf = df[~(df['title'].isin([np.nan]) | df['description'].isin([np.nan]))]
	for indexs in subdf.index:
		dict_title[subdf.loc[indexs]['asin']] = subdf.loc[indexs]['title']
		dict_description[subdf.loc[indexs]['asin']] = subdf.loc[indexs]['description']

	#### write generated dictionary into files
	rw.write2file(dict_title, foutput_title)
	rw.write2file(dict_description, foutput_description)
	print("Write Done!")
	print("Info: %d/%d"%(subdf.shape[0], df.shape[0]))