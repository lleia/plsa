#include "model.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int training()
{
	g_log_likelihood = 0.0;
	g_mdl_id = 0;
	g_steps_counter = 0;

	/* temporary vars */
	int tpc_idx = 0;
	int word_idx = 0;
	int doc_idx = 0;
	double denom = 0.0;
	double numer = 0.0;
	double log_likelihood = 0.0;
	double llf_temp = 0.0;
	int doc_size = 0;
	int wid = 0;

	char model_file_name[128];

	double * tpc_buf = (double *)malloc(g_tpc_count * sizeof(double));
	if (tpc_buf == NULL) {
		return -1;
	}

	while (1) {
		log_likelihood = 0.0;

		g_steps_counter ++;
		struct model * p_pre_model = &mdl[g_mdl_id];
		revert(g_mdl_id);
		struct model * p_now_model = &mdl[g_mdl_id];
		reset_model(p_now_model);

		double ** p_now_tpc = p_now_model->p_tpc_dis;
		double ** p_now_doc = p_now_model->p_doc_dis;
		double ** p_pre_tpc = p_pre_model->p_tpc_dis; 
		double ** p_pre_doc = p_pre_model->p_doc_dis;


		for (doc_idx = 0; doc_idx < g_doc_count; doc_idx++) {
			struct doc * pd = &p_doc_list[doc_idx];
			for (word_idx = 0; word_idx < pd->doc_size; word_idx ++) {

				struct word * pw = pd->word_list + word_idx;
				/* 
				 * E-step:
				 *
				 * Update the posterior probability:
				 *
				 *   p(z|d,w) = p(z|d)*p(w|z) / sigma_z' {p(z'|d)*p(w|z')}  
				 *
				 */

				denom = 0.0;
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					numer = p_pre_tpc[tpc_idx][pw->id] * p_pre_doc[doc_idx][tpc_idx];
					tpc_buf[tpc_idx] = numer;
					denom += numer;
				}

				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					tpc_buf[tpc_idx] /= denom;
					tpc_buf[tpc_idx] *= pw->weight;
					p_now_doc[doc_idx][tpc_idx] += tpc_buf[tpc_idx];
					p_now_tpc[tpc_idx][pw->id] += tpc_buf[tpc_idx];
				}
			}
		}

		/*
		 * M-step:
		 *
		 * Normalize the parameters:
		 * 
		 *   p(w|z) = sigma_d { f(d,w)*p(z|d,w) } / sigma_d_w { f(d,w)*p(z|d,w) }
		 *   p(z|d) = sigma_w { f(d,w)*p(z|d,w) } / sigma_w_z { f(d,w)*p(z|d,w) }
		 *
		 */

		for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
			denom = 0.0;
			for (word_idx = 0; word_idx < g_word_count; word_idx ++)
				denom += p_now_tpc[tpc_idx][word_idx];
			for (word_idx = 0; word_idx < g_word_count; word_idx ++)
				p_now_tpc[tpc_idx][word_idx] /= denom;
		}

		for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
			/*
			denom = 0.0;
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++)
				denom += p_now_doc[doc_idx][tpc_idx]; 
			*/
			for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++)
				p_now_doc[doc_idx][tpc_idx] /= p_doc_list[doc_idx].doc_weights; 
		}

		/* check convergence */
		for (doc_idx = 0; doc_idx < g_doc_count; doc_idx ++) {
			struct doc * pd = p_doc_list + doc_idx;
			doc_size = pd -> doc_size;
			for (word_idx = 0; word_idx < doc_size; word_idx ++) {
				llf_temp = 0.0;
				struct word * pw = pd->word_list + word_idx;
				wid = pw -> id;
				for (tpc_idx = 0; tpc_idx < g_tpc_count; tpc_idx ++) {
					llf_temp += p_now_doc[doc_idx][tpc_idx] * p_now_tpc[tpc_idx][wid];
				}
				if (llf_temp > EPSILON) {
					log_likelihood += pw->weight * log(llf_temp);
				}
			}
		}
	
		if (g_steps_counter >= g_save_thresh && (g_steps_counter - g_save_thresh) % 
				g_save_step == 0) {
			sprintf(model_file_name,"iter.%d.model",g_steps_counter);
			save_model(model_file_name);
		}

		fprintf(stderr,"Iteration [%d] : log_likelihood = %lf, g_log_likelihood = %lf\n",
				g_steps_counter,log_likelihood,g_log_likelihood);        
		if (g_steps_counter > 1 && fabs(log_likelihood - g_log_likelihood) < CONVERGENCE_THRESHOLD)
			break;
		if (g_max_steps_count > 0 && g_steps_counter >= g_max_steps_count)
			break;
		g_log_likelihood = log_likelihood;
	}	

	save_model("final.model");
	free(tpc_buf);

	return 0;
}
