#ifndef _PLSA_H_
#define _PLSA_H_

extern int g_tpc_count;
/* set model parameters */
extern void set_model_para(int doc_count, int word_count, int topic_count);
/* set trainer parameters */
extern void set_training_para(int max_step, int save_thresh, int save_step, int threads_num);

/* malloc memory for the model */
extern int malloc_model();
/* get a random model to train */
extern int init_random_model();  

/* load a trained model */
extern int load_model(const char * p_model_file);
/* load training file */
extern int load_training_file(const char * p_training_file);

/* start training procedure */
extern int training();
/* mulithreads version of training */
extern int parallel_training();
/* fold in of unseen document */
extern int fold_in(const char * p_doc, double * p_tpc_dis);

/* save the model in text file */
extern void save_text_model(const char * tpc_file, const char * doc_file, const char * info_file);
/* save the model in binary format */
extern int save_model(const char * p_model_file);

/* release the model */
extern void release();

#endif
