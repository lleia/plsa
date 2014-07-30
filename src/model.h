#ifndef _PLSA_MODEL_H_
#define _PLSA_MODEL_H_

#define DEBUG
#define revert(x) (x^=0x1)

#ifdef DEBUG
	#define dlog(...) fprintf(stderr,__VA_ARGS__)
#else
	#define dlog(...) 
#endif

struct model {
	/* p(w|z) */
	double ** p_tpc_dis;
	/* p(z|d) */
	double ** p_doc_dis;
};

struct word {
	int id;
	double weight;
};

struct doc {
	int doc_size;
	double doc_weights;
	struct word * word_list;
};

extern double CONVERGENCE_THRESHOLD;
extern double EPSILON;

extern int g_doc_count;
extern int g_tpc_count;
extern int g_word_count;
extern double g_log_likelihood;

extern int g_max_steps_count;
extern int g_steps_counter;
extern int g_save_step;
extern int g_save_thresh;
extern int g_threads_num;

extern int g_mdl_id;
extern struct model mdl[2];
extern struct doc * p_doc_list;

void reset_model(struct model * pm);
int save_model(const char * p_model_file);
int generate_distribution(double * prob, int count);

#endif
