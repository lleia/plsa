#include "model.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <pthread.h>

/* parallel variables */
static pthread_attr_t attr;
static pthread_barrier_t barr;
static pthread_t * p_threads_list = NULL;
static pthread_mutex_t * p_mutex_list = NULL;
/* a barrier used to check convergence and reset variables */
static pthread_mutex_t conv_mutex;
static pthread_cond_t conv_cond;
static int arrived_threads_count = 0;

/* used to calculste posterior probability */
static double ** p_thr_buf;

/* data blocks for every thread */
struct job {
	int thr_id;
	int doc_begin;
	int doc_end;
	int tpc_begin;
	int tpc_end;
};
static struct job * p_job_list = NULL;

/* buffer used to check convergence */
static double * p_conv_buf = NULL;
/* 1 when convergence */
static int conv_flag = 0;

/* TEMPORARY variables */
static struct model * p_pre_model = NULL;
static struct model * p_now_model = NULL; 
static double ** p_now_tpc = NULL; 
static double ** p_now_doc = NULL;
static double ** p_pre_tpc = NULL; 
static double ** p_pre_doc = NULL; 
static double log_likelihood = 0.0;
static char model_file_name[128];

void destroy_all()
{
	if (p_threads_list != NULL) {
		free(p_threads_list);
		p_threads_list = NULL;
	}

	if (p_mutex_list != NULL) {
		free(p_mutex_list);
		p_mutex_list = NULL;
	}

	if (p_thr_buf != NULL) {
		int thr_id = 0;
		for ( ; thr_id < g_threads_num; thr_id ++) {
			if (p_thr_buf[thr_id] != NULL) {
				free(p_thr_buf[thr_id]);
			}
		}
		free(p_thr_buf);
		p_thr_buf = NULL;
	}

	if (p_job_list != NULL) {
		free(p_job_list);
		p_job_list = NULL;
	}

	if (p_conv_buf != NULL) {
		free(p_conv_buf);
		p_conv_buf = NULL;
	}

	pthread_attr_destroy(&attr);
	pthread_barrier_destroy(&barr);
}

int init_all()
{
	fprintf(stderr,"start to initialize local variables\n");
	/* TEMPORARY variables */
	int thr_id;
	int job_id;

	if (g_threads_num <= 1 || g_threads_num > g_tpc_count || g_threads_num > g_doc_count) {
		fprintf(stderr, "Illegal mutithread configurations, g_threads_num = %d, g_tpc_count = %d, g_doc_count = %d\n",
				g_threads_num,g_tpc_count,g_doc_count);
		return -1;
	}

	p_threads_list = malloc(g_threads_num * sizeof(pthread_t));
	if (p_threads_list == NULL) {
		fprintf(stderr,"Create threads list error\n");
		destroy_all();
		return -1;
	}

	p_mutex_list = malloc(g_word_count * sizeof(pthread_mutex_t));
	if (p_mutex_list == NULL) {
		fprintf(stderr,"Create mutex list error\n");
		destroy_all();
		return -1;
	}

	p_thr_buf = malloc(g_threads_num * sizeof(double *));
	if (p_thr_buf == NULL) {
		fprintf(stderr,"Create thread buffer error\n");
		destroy_all();
		return -1;
	}

	for (thr_id = 0; thr_id < g_threads_num; thr_id ++) {
		p_thr_buf[thr_id] = malloc(g_tpc_count * sizeof(double));
		if (p_thr_buf[thr_id] == NULL) {
			fprintf(stderr,"Create thread buffer error\n");
			destroy_all();
			return -1;
		}
	}

	p_job_list = malloc(g_threads_num * sizeof(struct job));
	if (p_job_list == NULL) {
		fprintf(stderr,"Create job list error\n");
		destroy_all();
		return -1;
	}

	int doc_unit = g_doc_count / g_threads_num;
	int tpc_unit = g_tpc_count / g_threads_num;
	for (job_id = 0; job_id < g_threads_num; job_id ++) {
		struct job * pj = p_job_list + job_id;
		pj -> thr_id = job_id;
		pj -> doc_begin = job_id * doc_unit;
	   	pj -> doc_end = job_id == g_threads_num - 1 ? g_doc_count : (job_id + 1)*doc_unit;	
		pj -> tpc_begin = job_id * tpc_unit;
		pj -> tpc_end = job_id == g_threads_num - 1 ? g_tpc_count : (job_id + 1)*tpc_unit;
	}

	p_conv_buf = malloc(g_threads_num * sizeof(double));
	if (p_conv_buf == NULL) {
		fprintf(stderr,"Create convergence buffer error\n");
		destroy_all();
		return -1;
	}

	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr,PTHREAD_SCOPE_SYSTEM);
	pthread_barrier_init(&barr,NULL,g_threads_num);

	pthread_mutex_init(&conv_mutex,NULL);
	pthread_cond_init(&conv_cond,NULL);
	arrived_threads_count = 0;
	
	g_log_likelihood = 0.0;
	g_mdl_id = 0;
	g_steps_counter = 1;

	p_pre_model = &mdl[g_mdl_id];
	revert(g_mdl_id);
	p_now_model = &mdl[g_mdl_id];
	reset_model(p_now_model);

	p_now_tpc = p_now_model->p_tpc_dis;
	p_now_doc = p_now_model->p_doc_dis;
	p_pre_tpc = p_pre_model->p_tpc_dis; 
	p_pre_doc = p_pre_model->p_doc_dis;

	return 0;
}

void * training_worker(void * para)
{
	struct job * pj = (struct job *) para;
	int worker_id = pj -> thr_id;
	int doc_begin = pj -> doc_begin;
	int doc_end = pj -> doc_end;
	int tpc_begin = pj -> tpc_begin;
	int tpc_end = pj -> tpc_end;
	double * tpc_buf = p_thr_buf[worker_id];

	/* temporary vars */
	int tpc_idx = 0;
	int word_idx = 0;
	int doc_idx = 0;
	double denom = 0.0;
	double numer = 0.0;
	double llf_temp = 0.0;
	int thr_id = 0;
	int wid = 0;
	int doc_size = 0;

	while (1) {
		for (doc_idx = doc_begin; doc_idx < doc_end; doc_idx++) {
			struct doc * pd = &p_doc_list[doc_idx];
			for (word_idx = 0; word_idx < pd->doc_size; word_idx ++) {
				struct word * pw = pd->word_list + word_idx;
				wid = pw -> id; 
				/* 
				 * E-step:
				 * Update the posterior probability:
				 *   p(z|d,w) = p(z|d)*p(w|z) / sigma_z' {p(z'|d)*p(w|z')}  
				 */

				denom = 0.0;
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					numer = p_pre_tpc[tpc_idx][wid] * p_pre_doc[doc_idx][tpc_idx];
					tpc_buf[tpc_idx] = numer;
					denom += numer;
				}

				/* update doc-topic distribution */
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					tpc_buf[tpc_idx] /= denom;
					tpc_buf[tpc_idx] *= pw->weight;
					p_now_doc[doc_idx][tpc_idx] += tpc_buf[tpc_idx];
				}
				
				/* update topic-word distribution */
				pthread_mutex_lock(&p_mutex_list[wid]);
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					p_now_tpc[tpc_idx][wid] += tpc_buf[tpc_idx];
				}
				pthread_mutex_unlock(&p_mutex_list[wid]);
			}
		}

		pthread_barrier_wait(&barr);

		/*
		 * M-step:
		 *
		 * Normalize the parameters:
		 * 
		 *   p(w|z) = sigma_d { f(d,w)*p(z|d,w) } / sigma_d_w { f(d,w)*p(z|d,w) }
		 *   p(z|d) = sigma_w { f(d,w)*p(z|d,w) } / sigma_w_z { f(d,w)*p(z|d,w) }
		 *
		 */
		for (tpc_idx = tpc_begin; tpc_idx < tpc_end; tpc_idx ++) {
			denom = 0.0;
			for (word_idx = 0; word_idx < g_word_count; word_idx ++)
				denom += p_now_tpc[tpc_idx][word_idx];
			for (word_idx = 0; word_idx < g_word_count; word_idx ++)
				p_now_tpc[tpc_idx][word_idx] /= denom;
		}

		for (doc_idx = doc_begin; doc_idx < doc_end; doc_idx ++) {
			/*
			denom = 0.0;
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++)
				denom += p_now_doc[doc_idx][tpc_idx]; 
			*/
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++)
				p_now_doc[doc_idx][tpc_idx] /= p_doc_list[doc_idx].doc_weights; 
		}

		/*
		 *  check convergence 
		 */
		pthread_barrier_wait(&barr);

		p_conv_buf[worker_id] = 0.0;
		for (doc_idx = doc_begin; doc_idx < doc_end; doc_idx ++) {
			struct doc * pd = p_doc_list + doc_idx;
			doc_size = pd -> doc_size;
			for (word_idx = 0; word_idx < doc_size; word_idx ++) {
				llf_temp = 0.0;
				struct word * pw = pd -> word_list + word_idx;
				wid = pw -> id;
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					llf_temp += p_now_doc[doc_idx][tpc_idx] * p_now_tpc[tpc_idx][wid]; 
				}
				if (llf_temp > EPSILON) {
					p_conv_buf[worker_id] += pw->weight * log(llf_temp);
				}
			}
		}

		pthread_mutex_lock(&conv_mutex);
		arrived_threads_count ++;
		if (arrived_threads_count < g_threads_num) {
			pthread_cond_wait(&conv_cond,&conv_mutex);
		} else {
			/* 
			 * I'm the last in this iteration and I check convgence 
			 * and reset iterative variables 
			 */
			log_likelihood = 0.0;
			for (thr_id = 0; thr_id < g_threads_num; thr_id ++) {
				log_likelihood += p_conv_buf[thr_id];
			}

			fprintf(stderr,"Iteration [%d] : log_likelihood = %lf, g_log_likelihood = %lf\n",
					g_steps_counter,log_likelihood,g_log_likelihood);        
			if (g_steps_counter > 1 && fabs(log_likelihood - g_log_likelihood) < CONVERGENCE_THRESHOLD) {
				fprintf(stderr,"The convergence threshold is satisfied,quit\n");
				conv_flag = 1;
			}

			if (g_max_steps_count > 0 && g_steps_counter >= g_max_steps_count) {
				fprintf(stderr,"Get the maximum iteration threshold, quit\n");
				conv_flag = 1;
			}

			g_log_likelihood = log_likelihood;

			/* save model */
			if (g_steps_counter >= g_save_thresh && (g_steps_counter - g_save_thresh) % g_save_step == 0) {
				sprintf(model_file_name,"iter.%d.model",g_steps_counter);
				save_model(model_file_name);
			}

			/* reset variables */
			if (conv_flag == 0) {
				g_steps_counter ++;
				arrived_threads_count = 0;

				p_pre_model = &mdl[g_mdl_id];
				revert(g_mdl_id);
				p_now_model = &mdl[g_mdl_id];
				reset_model(p_now_model);

				p_now_tpc = p_now_model->p_tpc_dis;
				p_now_doc = p_now_model->p_doc_dis;
				p_pre_tpc = p_pre_model->p_tpc_dis; 
				p_pre_doc = p_pre_model->p_doc_dis;
			}
			pthread_cond_broadcast(&conv_cond);
		}
		pthread_mutex_unlock(&conv_mutex);

		if (conv_flag == 1) {
			break;
		}
	}	

	save_model("final.model");
	return 0;
}

int parallel_training()
{
	if (init_all() != 0) {
		fprintf(stderr,"Initalize the parallel plsa error\n");
		return -1;
	}

	int thr_id = 0;
	for ( ;thr_id < g_threads_num; thr_id ++) {
		pthread_create(&p_threads_list[thr_id],&attr,training_worker,(void *) &p_job_list[thr_id]);
	}

	for (thr_id = 0; thr_id < g_threads_num; thr_id ++) {
		pthread_join(p_threads_list[thr_id],NULL);
	}

	destroy_all();
	return 0;
}
