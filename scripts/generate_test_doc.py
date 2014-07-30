import os
import sys

term_dict = {}

def load_term_dict(term_file):
	infp = open(term_file,'r')
	for line in infp:
		info = line.rstrip().split(' ')
		if len(info) != 2:
			print >> sys.stderr, 'Error in term file line = [%s]'%(line.rstrip())
			return False 
		term_dict[info[0]] = info[1]
	infp.close()
	return True

def generate_test_doc(infile,outfile):
	infp = open(infile,'r')
	outfp = open(outfile,'w')
	for line in infp:
		info = line.rstrip().split(' ')
		out_line = ''
		doc_dict = {}
		for term in info:
			if term in doc_dict:
				doc_dict[term] += 1
			else:
				doc_dict[term] = 1
		for pair in doc_dict.items():
			if pair[0] not in term_dict:
				continue
			if out_line != '':
				out_line += ' '
			out_line += '%s:%d'%(term_dict[pair[0]],pair[1])
		outfp.write(out_line + '\n')
	infp.close()
	outfp.close()

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print >> sys.stderr, 'Usage : %s [term file] [input file] [output file]'%(sys.argv[0])
		sys.exit(0)
	load_term_dict(sys.argv[1])
	generate_test_doc(sys.argv[2],sys.argv[3])
