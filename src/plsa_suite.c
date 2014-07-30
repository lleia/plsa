#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "plsa.h"

const int MAX_BUF_SIZE = 40960;

enum mode {
	TRAIN = 0,
	TEST,
	RETRAIN
};

void trim(char * buf)
{
	while (*buf && *buf != '\n') buf ++;
	*buf = '\0';
}

void usage()
{
	printf("   Usage : plsa [options] ...\n\
		-h : help\n\
		-t : training mode\n\
		-e : test mode\n\
		-r : retraining mode\n\
		-d : document count\n\
		-w : word count\n\
		-o : topic count\n\
		-p : threads count, 1 by default\n\
		-m : maximum iteration count, 1000 by default\n\
		-s : save threshold, 800 by default\n\
		-v : save step, 100 by default\n\
		-a : model file\n\
		-b : input file\n\
		-c : output file\n");	
}

int main(int argc, char ** argv)
{
	int doc_count = 0;
	int word_count = 0;
	int topic_count = 0;
	int threads_num = 1;
	int max_iters_count = 1000;
	int save_thresh = 800;
	int save_step = 100;
	enum mode mode_flag = TRAIN;
	const char * p_model_file = NULL;
	const char * p_input_file = NULL;
	const char * p_output_file = NULL;
	char iobuf[MAX_BUF_SIZE];
	int ch,tpc_idx;

	if (argc <= 2) {
		usage();
		return 0;
	}

	while ((ch = getopt(argc,argv,"hterd:w:o:p:m:s:v:a:b:c:")) != -1) {
		switch (ch) {
			case 'h':
				usage();
				return 0;
			case 't':
				mode_flag = TRAIN;
				break;
			case 'e':
				mode_flag = TEST;
				break;
			case 'r':
				mode_flag = RETRAIN;
				break;
			case 'd':
				sscanf(optarg,"%d",&doc_count);
				break;
			case 'w':
				sscanf(optarg,"%d",&word_count);
				break;
			case 'o':
				sscanf(optarg,"%d",&topic_count);
				break;
			case 'p':
				sscanf(optarg,"%d",&threads_num);
				break;
			case 'm':
				sscanf(optarg,"%d",&max_iters_count);
				break;
			case 's':
				sscanf(optarg,"%d",&save_thresh);
				break;
			case 'v':
				sscanf(optarg,"%d",&save_step);
				break;
			case 'a':
				p_model_file = optarg;
				break;
			case 'b':
				p_input_file = optarg;
				break;
			case 'c':
				p_output_file = optarg;
				break;
			default:
				usage();
				return 0;
		}
	}

	switch (mode_flag) {

		case TRAIN:

			if (p_input_file == NULL) {
				fprintf(stderr,"Illegal training file!\n");
				break;
			}

			if (doc_count <= 0 || word_count <= 0 || topic_count <= 0) {
				fprintf(stderr,"Illegal model parameters doc [%d], word [%d], topic [%d]\n",
						doc_count,word_count,topic_count);
				break;
			}

			if (max_iters_count <= 0 || save_thresh <= 0 || save_step <= 0) {
				fprintf(stderr,"Illegal training parameters, max [%d], save threshold [%d],save step [%d], threads count [%d]\n",
						max_iters_count,save_thresh,save_step,threads_num);
				break;
			}

			set_model_para(doc_count,word_count,topic_count);
			set_training_para(max_iters_count,save_thresh,save_step,threads_num);

			if (malloc_model() != 0) {
				fprintf(stderr,"Malloc memory for new model error\n");
				break;
			}

			load_training_file(p_input_file);
			init_random_model();

			if (threads_num == 1) {
				training();
			} else {
				parallel_training();
			}

			break;

		case RETRAIN:

			if (p_input_file == NULL) {
				fprintf(stderr,"Illegal training file!\n");
				break;
			}

			if (p_model_file == NULL) {
				fprintf(stderr,"Illegal model file!\n");
				break;
			}

			if (doc_count <= 0 || word_count <= 0 || topic_count <= 0) {
				fprintf(stderr,"Illegal model parameters doc [%d], word [%d], topic [%d]\n",
						doc_count,word_count,topic_count);
				break;
			}

			if (max_iters_count <= 0 || save_thresh <= 0 || save_step <= 0) {
				fprintf(stderr,"Illegal training parameters, max [%d], save threshold [%d], \
					save step [%d], threads count [%d]\n",max_iters_count,save_thresh,save_step,threads_num);
				break;
			}

			set_model_para(doc_count,word_count,topic_count);
			set_training_para(max_iters_count,save_thresh,save_step,threads_num);
			if (load_model(p_model_file) != 0) {
				fprintf(stderr,"Loaded model failed, quit...\n");
				break;
			}

			if (threads_num == 1) {
				training();
			} else {
				parallel_training();
			}
				
			break;

		case TEST:

			if (p_input_file == NULL) {
				fprintf(stderr,"Illegal training file!\n");
				break;
			}

			if (p_output_file == NULL) {
				fprintf(stderr,"Illegal output file!\n");
				break;
			}

			if (p_model_file == NULL) {
				fprintf(stderr,"Illegal model file!\n");
				break;
			}

			if (load_model(p_model_file) != 0) {
				fprintf(stderr,"Load model failed, quit\n");
				break;
			}

			FILE * infp = fopen(p_input_file,"r");
			FILE * outfp = fopen(p_output_file,"w");

			if (!infp || !outfp) {
				fprintf(stderr,"Open files error,quit\n");
				break;
			}

			double * p_doc_tpc_dis = malloc(g_tpc_count * sizeof(double));
			if (p_doc_tpc_dis == NULL) {
				fprintf(stderr,"Create memory for folding document error!\n");
				break;
			}

			while (fgets(iobuf,sizeof(iobuf),infp) != NULL) {
				trim(iobuf);
				fold_in(iobuf,p_doc_tpc_dis);
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++)
				{
					if (tpc_idx != 0)
						fprintf(outfp," ");
					fprintf(outfp,"%lf",p_doc_tpc_dis[tpc_idx]);
				}
				fprintf(outfp,"\n");
			}

			free(p_doc_tpc_dis);
			fclose(infp);
			fclose(outfp);
			break;
	}

	release();
	return 0;
}
