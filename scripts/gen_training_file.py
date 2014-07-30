import os
import sys

def generate(infile,word_id_file,training_file):
	term_dict = {}

	infp = open(infile,'r')
	for line in infp:
		info = line.rstrip().split(' ')
		for term in info:
			if term in term_dict:
				term_dict[term] += 1
			else:
				term_dict[term] = 1
	term_list = sorted(term_dict.items(),key=lambda x : x[1],reverse=True)	

	term_id = 0
	for term in term_list:
		term_dict[term[0]] = term_id
		term_id += 1

	infp.seek(0,0)
	outfp = open(training_file,'w')

	for line in infp:
		term_tf = {}
		info = line.rstrip().split(' ')
		for term in info:
			if term in term_tf:
				term_tf[term] += 1
			else:
				term_tf[term] = 1
		out_line = '' 
		for pair in term_tf.items():
			if out_line != '':
				out_line += ' '
			out_line += '%d:%d'%(term_dict[pair[0]],pair[1])
		outfp.write(out_line+'\n')

	infp.close()
	outfp.close()

	outfp = open(word_id_file,'w')
	for term in term_dict.items():
		outfp.write('%s %d\n'%(term[0],term[1]))
	outfp.close()

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print >> sys.stderr,'Usage : [%s] [data file] [word id file] [training file]'%(sys.argv[0])
		sys.exit(0)
	generate(sys.argv[1],sys.argv[2],sys.argv[3])
