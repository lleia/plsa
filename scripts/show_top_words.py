import os
import sys
import struct
import array

doc_count = 0
tpc_count = 0
word_count = 0

steps_counter = 0
log_likelihood = 0.0

tpc_dis = []
doc_dis = []

term_dict = {}

def load_model(model_file):
	global doc_count
	global tpc_count
	global word_count
	global steps_counter
	global log_likelihood
	infp = open(model_file,'rb')
	doc_count = struct.unpack('i',infp.read(4))[0]
	tpc_count = struct.unpack('i',infp.read(4))[0]
	word_count = struct.unpack('i',infp.read(4))[0]
	for tpx_id in range(0,tpc_count):
		tds = array.array('d')
		tds.fromfile(infp,word_count)
		tpc_dis.append(list(tds))
	for doc_id in range(0,doc_count):
		tds = array.array('d')
		tds.fromfile(infp,tpc_count)
		doc_dis.append(list(tds))
	steps_counter = struct.unpack('i',infp.read(4))
	log_likelihood = struct.unpack('d',infp.read(8))
	infp.close()

def load_term_dict(term_file):
	infp = open(term_file,'r')
	for line in infp:
		info = line.rstrip().split(' ')
		term_dict[int(info[1])] = info[0]
	infp.close()

def show_top_words(outfile,count):
	outfp = open(outfile,'w')
	for tid in range(0,tpc_count):
		tds = tpc_dis[tid]
		ftds = map(None,range(0,word_count),tds)
		ftds.sort(key=lambda x:x[1],reverse=True)	
		outfp.write('Topic %dth:\n'%(tid))
		for wid in range(0,count):
			outfp.write('    %s    %f\n'%(term_dict[ftds[wid][0]],ftds[wid][1]))
	outfp.close()

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print >> sys.stderr, 'Usage : [%s] [model file] [term file] [output file] [count]'%(sys.argv[0])
		sys.exit(0)
	load_model(sys.argv[1])
	load_term_dict(sys.argv[2])
	print >> sys.stderr, 'doc_count = %d'%(doc_count)
	print >> sys.stderr, 'tpc_count = %d'%(tpc_count)
	print >> sys.stderr, 'word_count = %d'%(word_count)
	print >> sys.stderr, 'step_counter = %d'%(steps_counter)
	print >> sys.stderr, 'log_likelihood = %f'%(log_likelihood)
	show_top_words(sys.argv[3],int(sys.argv[4]))
