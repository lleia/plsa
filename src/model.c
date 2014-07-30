#include "model.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

double CONVERGENCE_THRESHOLD = 0.001;
double EPSILON = 0.0000001;

static const int MAX_LINE_SIZE = 40960;

/* data */
int g_doc_count = 0;
int g_tpc_count = 0;
int g_word_count = 0;
int g_max_steps_count = 0;
int g_save_thresh = 0;
int g_save_step = 0;

struct doc * p_doc_list = NULL;

/* mdl indicator */
int g_mdl_id = 0;
/* model */
struct model mdl[2];

/* iteration counts */
int g_steps_counter = 0;
/* number of trainer threads */
int g_threads_num = 1;
/* log likelihood of em */
double g_log_likelihood = 0.0;

void set_model_para(int doc_count, int word_count, int topic_count)
{
	g_doc_count = doc_count;
	g_word_count = word_count;
	g_tpc_count = topic_count;
}

void set_training_para(int max_step, int save_thresh, int save_step, int threads_num)
{
	g_max_steps_count = max_step;
	g_save_thresh = save_thresh;
	g_save_step = save_step;
	g_threads_num = threads_num;
}

void reset_model(struct model * pm)
{
	int doc_idx,tpc_idx,word_idx;
	for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
			pm->p_doc_dis[doc_idx][tpc_idx] = 0.0;
		}
	}

	for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
		for (word_idx = 0; word_idx < g_word_count; word_idx ++) {
			pm->p_tpc_dis[tpc_idx][word_idx] = 0.0;
		}
	}
}

/* generate a simple distribution*/
int generate_distribution(double * prob,int count)
{
	if (prob == NULL || count < 0)
		return -1;
	double sum = 0.0;
	int id = 0;
	for ( ; id < count; id ++) {
		double item = rand()%1024;
		sum += item;
		prob[id] = item;
	}

	for (id = 0; id < count; id++) {
		prob[id] /= sum;
	}
	return 0;
}

int load_training_file(const char * p_training_file)
{
	int doc_idx, word_idx;
	if (g_doc_count <= 0 || g_word_count <= 0 || g_tpc_count <= 0) {
		fprintf(stderr,"Model parameters should be initialized!\n");
		return -1;
	}

	/* create doc list memeory */
	p_doc_list = malloc(g_doc_count * sizeof(struct doc)); 
	if (p_doc_list == NULL) {
		return -1;
	}

	/* open the file */
	FILE * infp = fopen(p_training_file,"r");
	if (infp == NULL) {
		fprintf(stderr,"Can't open training file [%s]\n",p_training_file);
		return -1;
	}

	doc_idx = 0;
	char buf[MAX_LINE_SIZE];
	while (fgets(buf,sizeof(buf),infp)) {
		if (doc_idx >= g_doc_count) {
			fprintf(stderr,"Documents count error while doc_idx = [%d] while g_doc_count = [%d]!\n", doc_idx, g_doc_count);
			fclose(infp);
			return -1;
		}
		/* create doc information memory */
		int doc_size = 1;
		char * pb = buf;
		while (*pb) {
			if (*pb == ' ') doc_size ++;
			pb ++;
		}

		p_doc_list[doc_idx].doc_weights = 0.0;
		p_doc_list[doc_idx].doc_size = doc_size;
		p_doc_list[doc_idx].word_list = malloc(doc_size * sizeof(struct word));
		if (p_doc_list[doc_idx].word_list == NULL) {
			fprintf(stderr,"Create memory for word list of the %dth document error!\n",
					doc_idx);
			fclose(infp);
			return -1;
		}

		/* create word information memory and load them from file */
		word_idx = 0;
		pb = buf;
		struct word * p_word_list = p_doc_list[doc_idx].word_list;
		while (*pb) {
			if (*pb == ' ') pb++;
			sscanf(pb,"%d:%lf",&(p_word_list[word_idx].id),
					&(p_word_list[word_idx].weight)); 
			p_doc_list[doc_idx].doc_weights += p_word_list[word_idx].weight;
			if (p_word_list[word_idx].id >= g_word_count) {
				fprintf(stderr,"Get illegal word id [%d] while g_word_count = [%d]\n",
						p_word_list[word_idx].id,g_word_count);
				return -1;
			}
			word_idx ++;
			while (*pb && *pb != ' ') pb ++;
		}
		doc_idx ++;
	}

	return 0;
}

int malloc_model()
{
	if (g_doc_count <= 0 || g_word_count <= 0 || g_tpc_count <= 0) {
		fprintf(stderr,"Model parameters should be initialized!\n");
		return -1;
	}

	int doc_idx, tpc_idx, mmid;
	/* create model memory */
	for (mmid = 0; mmid < 2; mmid ++) {
		struct model * pm = &mdl[mmid];
		pm -> p_tpc_dis = malloc(g_tpc_count * sizeof(double *));	
		if (pm -> p_tpc_dis == NULL) {
			fprintf(stderr,"Create memory for the topic distribution error!\n");
			return -1;
		}

		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
			pm -> p_tpc_dis[tpc_idx] = malloc(g_word_count * sizeof(double));
			if (pm -> p_tpc_dis[tpc_idx] == NULL) {
				fprintf(stderr,"Create memory for the %dth topic distribution error!\n",
						tpc_idx);
				return -1;
			}
		}

		pm -> p_doc_dis = malloc(g_doc_count * sizeof(double *));
		if (pm -> p_doc_dis == NULL) {
			fprintf(stderr,"Create memory for document distribution error!\n");
			return -1;
		}

		for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
			pm -> p_doc_dis[doc_idx] = malloc(g_tpc_count * sizeof(double));
			if (pm -> p_doc_dis[doc_idx] == NULL) {
				fprintf(stderr,"Create memory for the %dth document distribution error!\n"
						,doc_idx);
				return -1;
			}
		}
	}
	return 0;
}

int init_random_model()
{
	if (g_tpc_count <= 0 || g_doc_count <= 0 || g_word_count <= 0) {
		fprintf(stderr,"Model parameters should be initialized!\n");
		return -1;
	}

	int tpc_idx, doc_idx;
	struct model * pm = &mdl[g_mdl_id];
	for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
		generate_distribution(pm->p_tpc_dis[tpc_idx],g_word_count);
	}

	for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
		generate_distribution(pm->p_doc_dis[doc_idx],g_tpc_count);
	}

	return 0;
}

int save_model(const char * p_model_file)
{
	/* temporary index vars */
	int tpc_idx = 0;
	int doc_idx = 0;

	FILE * model_fp = fopen(p_model_file,"wb");
	if (model_fp == NULL) {
		fprintf(stderr,"Open file [%s] failed!\n",p_model_file);
		fclose(model_fp);
		return -1;
	}

	if (fwrite(&g_doc_count,sizeof(g_doc_count),1,model_fp) != 1) {
		fprintf(stderr,"Write document count error!\n");
		fclose(model_fp);
		return -1;
	}	

	if (fwrite(&g_tpc_count,sizeof(g_tpc_count),1,model_fp) != 1) {
		fprintf(stderr,"Write topic count error!\n");
		fclose(model_fp);
		return -1;
	}

	if (fwrite(&g_word_count,sizeof(g_word_count),1,model_fp) != 1) {
		fprintf(stderr,"Write word count error!\n");
		fclose(model_fp);
		return -1;
	}

	/* save topic distribution */
	double ** p_tpc_dis = mdl[g_mdl_id].p_tpc_dis;
	for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
		if (fwrite(p_tpc_dis[tpc_idx],sizeof(double),g_word_count,model_fp) != 
				g_word_count) {
			fprintf(stderr,"Save the %dth topic distribution error!\n",tpc_idx);
			fclose(model_fp);
			return -1;
		}
	}

	/* save document distribution */
	double ** p_doc_dis = mdl[g_mdl_id].p_doc_dis;
	for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
		if (fwrite(p_doc_dis[doc_idx],sizeof(double),g_tpc_count,model_fp) != 
				g_tpc_count) {
			fprintf(stderr,"Save the %dth document distribution error!\n",doc_idx);
			fclose(model_fp);
			return -1;
		}
	}

	if (fwrite(&g_steps_counter,sizeof(g_steps_counter),1,model_fp) != 1) {
		fprintf(stderr,"Write iteration steps error!\n");
		fclose(model_fp);
		return -1;
	}

	if (fwrite(&g_log_likelihood,sizeof(g_log_likelihood),1,model_fp) != 1) {
		fprintf(stderr,"Write log likelihood to file error!\n");
		fclose(model_fp);
		return -1;
	}

	fclose(model_fp);
	return 0;
}

int load_model(const char * p_model_file)
{
	FILE * model_fp = fopen(p_model_file,"rb");
	if (model_fp == NULL) {
		fprintf(stderr,"Open model file [%s] error!\n",p_model_file);
		return -1;
	}

	int para_flag = 0;
	if (g_doc_count == 0 && g_word_count == 0 && g_tpc_count == 0) {
		para_flag = 1;
	}

	/* read document count from model file */
	int doc_idx, tpc_idx, tmp_count = 0;
	if (fread(&tmp_count,sizeof(tmp_count),1,model_fp) != 1) {
		fprintf(stderr,"Read doc count from model error!\n");
		fclose(model_fp);
		return -1;
	}

	if (g_doc_count != 0 && g_doc_count != tmp_count) {
		fprintf(stderr,"Illegal document count, g_doc_count = %d versus %d\n",
				g_doc_count,tmp_count);
		fclose(model_fp);
		return -1;
	} else {
		g_doc_count = tmp_count;
	}

	/* read topic count from model file */
	if (fread(&tmp_count,sizeof(tmp_count),1,model_fp) != 1) {
		fprintf(stderr,"Read topic count from model error!\n");
		fclose(model_fp);
		return -1;
	}

	if (g_tpc_count != 0 && g_tpc_count != tmp_count) {
		fprintf(stderr,"Illegal document count, g_doc_count = %d versus %d\n",
				g_tpc_count,tmp_count);
		fclose(model_fp);
		return -1;
	} else {
		g_tpc_count = tmp_count;
	}

	/* read word count from model file */
	if (fread(&tmp_count,sizeof(tmp_count),1,model_fp) != 1) {
		fprintf(stderr,"Read topic count from model error!\n");
		fclose(model_fp);
		return -1;
	}

	if (g_word_count != 0 && g_word_count != tmp_count) {
		fprintf(stderr,"Illegal document count, g_doc_count = %d versus %d\n",
				g_word_count,tmp_count);
		fclose(model_fp);
		return -1;
	} else {
		g_word_count = tmp_count;
	}

	if (para_flag == 1) {
		malloc_model();
	}

	/* load topic distribution */
	double ** p_tpc_dis = mdl[g_mdl_id].p_tpc_dis;
	for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
		if (fread(p_tpc_dis[tpc_idx],sizeof(double),g_word_count,model_fp) != 
				g_word_count) {
			fprintf(stderr,"Read the %dth topic distribution error!\n",tpc_idx);
			fclose(model_fp);
			return -1;
		}
	}

	/* load document distribution */
	double ** p_doc_dis = mdl[g_mdl_id].p_doc_dis;
	for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
		if (fread(p_doc_dis[doc_idx],sizeof(double),g_tpc_count,model_fp) != 
				g_tpc_count) {
			fprintf(stderr,"Read the %dth document distribution error!\n",doc_idx);
			fclose(model_fp);
			return -1;
		}
	}

	if (fread(&g_steps_counter,sizeof(g_steps_counter),1,model_fp) != 1) {
		fprintf(stderr,"Write iteration steps error!\n");
		fclose(model_fp);
		return -1;
	}

	if (fread(&g_log_likelihood,sizeof(g_log_likelihood),1,model_fp) != 1) {
		fprintf(stderr,"Write log likelihood to file error!\n");
		fclose(model_fp);
		return -1;
	}

	fclose(model_fp);
	return 0;
}

void release()
{
	/* temporary index vars */
	int tpc_idx = 0;
	int doc_idx = 0;
	int mmid = 0;

	if (p_doc_list != NULL) {
		for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
			if (p_doc_list[doc_idx].word_list != NULL) {
				free(p_doc_list[doc_idx].word_list);
			}
		}
		free(p_doc_list);
		p_doc_list = NULL;
	}

	for (mmid = 0 ; mmid < 2; mmid ++) {
		struct model * pm = &mdl[mmid];
		if (pm -> p_tpc_dis != NULL) {
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
				if (pm->p_tpc_dis[tpc_idx] != NULL) {
					free(pm->p_tpc_dis[tpc_idx]);
					pm->p_tpc_dis[tpc_idx] = NULL;
				}
			}
			free(pm->p_tpc_dis);
			pm->p_tpc_dis = NULL;
		}
		if (pm -> p_doc_dis != NULL) {
			for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
				if (pm->p_doc_dis[doc_idx] != NULL) {
					free(pm->p_doc_dis[doc_idx]);
					pm->p_doc_dis[doc_idx] = NULL;
				}
			}
			free(pm->p_doc_dis);
			pm->p_doc_dis = NULL;
		}
	}

	g_doc_count = 0;
	g_tpc_count = 0;
	g_word_count = 0;
}

void save_text_model(const char * tpc_file, const char * doc_file, const char * info_file)
{
	/* temporary index vars */
	int tpc_idx = 0;
	int word_idx = 0;
	int doc_idx = 0;
	
	FILE * infofp = fopen(info_file,"w");
	if (infofp == NULL) {
		return;
	}

	fprintf(infofp,"document count : %d\n",g_doc_count);
	fprintf(infofp,"word count : %d\n",g_word_count);
	fprintf(infofp,"topic count : %d\n",g_tpc_count);
	fprintf(infofp,"max iteration steps : %d\n",g_max_steps_count);
	fprintf(infofp,"iteration count : %d\n", g_steps_counter);
	fprintf(infofp,"log likelihood function : %lf\n", g_log_likelihood);
	fprintf(infofp,"threads num : %d\n",g_threads_num);
	fclose(infofp);

	struct model * pm = &mdl[g_mdl_id];
	FILE * tpcfp = fopen(tpc_file,"w");
	if (tpcfp == NULL) {
		return;
	}

	for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
		for (word_idx = 0; word_idx < g_word_count; word_idx ++) {
			if (word_idx != 0)
				fprintf(tpcfp," ");
			fprintf(tpcfp,"%lf", pm->p_tpc_dis[tpc_idx][word_idx]);
		}
		fprintf(tpcfp,"\n");
	}
	fclose(tpcfp);

	FILE * docfp = fopen(doc_file,"w");
	if (docfp == NULL) {
		return;
	}

	for ( doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx++) {
			if (tpc_idx != 0)
				fprintf(docfp," ");
			fprintf(docfp,"%lf",pm->p_doc_dis[doc_idx][tpc_idx]);
		}
		fprintf(docfp,"\n");
	}
	fclose(docfp);
}
