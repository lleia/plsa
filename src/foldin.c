#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model.h"

int fold_in(const char * p_doc, double * p_out_doc_dis) {

	int word_idx, tpc_idx;

	if (p_doc == NULL || p_out_doc_dis  == NULL) { 
		return -1;
	}

	/* create document struct */
	int doc_size = 1;
	const char * ptmp = p_doc;
	while (*ptmp) {
		if (*ptmp == ' ') doc_size ++;
		ptmp ++;
	}

	struct doc doc_item;
	doc_item.doc_size = doc_size;
	doc_item.doc_weights = 0.0;

	doc_item.word_list = malloc(doc_size * sizeof(struct word));
	if (doc_item.word_list == NULL) {
		fprintf(stderr,"Create memory for word list error!\n");
		return -1;
	}

	/* parse word information from string data */
	word_idx = 0;
	ptmp = p_doc;
	struct word * p_word_list = doc_item.word_list;
	while (*ptmp) {
		if (*ptmp == ' ') ptmp ++;
		sscanf(ptmp,"%d:%lf",&(p_word_list[word_idx].id),&(p_word_list[word_idx].weight));
		doc_item.doc_weights += p_word_list[word_idx].weight;

		if (p_word_list[word_idx].id >= g_word_count) {
			fprintf(stderr,"Get illegal word id [%d] while g_word_count = [%d]\n",
					p_word_list[word_idx].id,g_word_count);
			free(doc_item.word_list);
			return -1;
		}
		word_idx ++;
		while (*ptmp && *ptmp != ' ') ptmp ++;
	}

	/* create topic buffer and initialize a random model */
	/* TODO: initalize these variables outside the funtion */
	double * p_doc_tpc_buf[2];

	p_doc_tpc_buf[0] = malloc(g_tpc_count * sizeof(double));
	if (p_doc_tpc_buf[0] == NULL) {
		fprintf(stderr,"Create topic buffer memory error!\n");
		return -1;
	}
	p_doc_tpc_buf[1] = malloc(g_tpc_count * sizeof(double));
	if (p_doc_tpc_buf[1] == NULL) {
		fprintf(stderr,"Create topic buffer memory error!\n");
		return -1;
	}

	/* create posterior topic distribution */
	/* p(z|d,w) */
	double * p_post_tpc_buf = malloc(g_tpc_count * sizeof(double));
	if (p_post_tpc_buf == NULL) {
		fprintf(stderr,"Create posterior distribution error\n");
		return -1;
	}

	int iter_counter = 0;
	double numer,denom,llf_temp;
	double now_likelihood = 0.0; 
	double pre_likelihood = 0.0;
	/* p(w|z) which is const here */
	const double ** p_tpc_dis = (const double **)mdl[g_mdl_id].p_tpc_dis;
	/* p(z|d), update them in the below loops */
	int mdl_id = 1;
	double * p_pre_doc = p_doc_tpc_buf[0];
	double * p_now_doc = p_doc_tpc_buf[1];
	/* generate a random model to start */
	generate_distribution(p_pre_doc, g_tpc_count);
	
	while (1) {

		iter_counter ++;
		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
			p_now_doc[tpc_idx] = 0.0;
		}

		for (word_idx = 0; word_idx < doc_item.doc_size; word_idx ++) {
			struct word * pw = doc_item.word_list + word_idx;
			denom = 0.0;
			/* calculate posterior probability for p(z|d,w) */
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
				numer = p_pre_doc[tpc_idx] * p_tpc_dis[tpc_idx][pw->id]; 
				p_post_tpc_buf[tpc_idx] = numer;
				denom += numer;
			}

			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
				p_post_tpc_buf[tpc_idx] /= denom;
				p_now_doc[tpc_idx] += p_post_tpc_buf[tpc_idx] * pw->weight;
			}
		}

		/* normalized p(z|d), that is p_now_tpc */
		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
			p_now_doc[tpc_idx] /= doc_item.doc_weights;
		}

		/* check convergence */
		now_likelihood = 0.0;
		for (word_idx = 0; word_idx < doc_item.doc_size; word_idx ++) {
			llf_temp = 0.0;
			struct word * pw = doc_item.word_list + word_idx;
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
				llf_temp += p_now_doc[tpc_idx] * p_tpc_dis[tpc_idx][pw->id];
			}
			if (llf_temp > EPSILON) {
				now_likelihood += pw->weight * log(llf_temp);
			}
		}

		fprintf(stderr,"Iteration [%d]: now_likelihood = %lf\n",
				iter_counter, now_likelihood);
		if (iter_counter > 1 && fabs(now_likelihood - pre_likelihood) 
				< CONVERGENCE_THRESHOLD) {
			fprintf(stderr,"Cross the convergence threshold,quit\n");
			break;
		}
		/* revert variables */
		pre_likelihood = now_likelihood;
		p_pre_doc = p_doc_tpc_buf[mdl_id];	
		revert(mdl_id);
		p_now_doc = p_doc_tpc_buf[mdl_id];
	}

	memcpy((void*)p_out_doc_dis,(void*)p_now_doc,g_tpc_count*sizeof(double));

	free(doc_item.word_list);
	free(p_doc_tpc_buf[0]);
	free(p_doc_tpc_buf[1]);
	free(p_post_tpc_buf);
	return 0;
}
