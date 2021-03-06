import shelve
import pickle


def write2file(dct, path):
	with shelve.open(path, protocol=pickle.HIGHEST_PROTOCOL) as d:
		d['content'] = dct

def readffile(path):
	with shelve.open(path, protocol=pickle.HIGHEST_PROTOCOL) as d:
		return d['content']