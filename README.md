# SSBM - Self-supervised Seed-driven Bayesian-Modeling
This code repository is related to the paper *A Self-Supervised Seed-driven Approach for Topic Modelling and Clustering* by Ravenda et al. <br><br>
<img src="https://github.com/Fede-stack/SSBM-Self-supervised-Seed-driven-Bayesian-Modeling/blob/main/images/lillo.png" alt="" width="300">

# SSBM: Expert Recommendations for Hyperparameter Tuning

**SSBM** is a simple, lightweight, and intuitive model that's also very powerful for Topic Modelling and Clustering tasks (based on Topics) in a <span style="color: red;">self-supervised fashion</span>. Below are some *expert* recommendations for selecting hyperparameters if you wish to use it as an easy-to-go model:

- `N`: Empirically, the best results are obtained by selecting a value around `n_topics * 15`.

- `p`: A good selection of `p` depends on the vocabulary size and the value of `N`. Typically, values between 15 and 100 yield good results. If the vocabulary is limited and the number of topics is high, it's better to use a low value (10-15).

- `d`: A low value of *d* (e.g., 2) is advisable when the average document length is below 40-50. With longer documents, it's recommended to use a higher value.

- `c`: This controls the balance of labels generated by the model. A value between 3 and 8 is usually sufficient. For datasets with many observations, opt for lower values.

- `W`: `range(5, 15)` is usually a good choice. 

- `sel_mod`: Model selection. Three models are considered, but additional models can be easily integrated. Random Forest and Neural Network models generally perform best.

```python 
ssbm = SSBM(docs, n_topics = n_topics)
predictions, topic_representations, coherence_npmi, coherence_uci, coherence_cv, coherence_div, prediction_clusters = ssbm.train(N=N, p=p, d=d, W=W, c=c, sel_mod=sel_mod)
ssbm.plot_topics(topic_representations)
```

# TP: TopicPropagation

<img src="https://github.com/Fede-stack/SSBM-Self-supervised-Seed-driven-Bayesian-Modeling/blob/main/images/otter.png" alt="" width="300">

To combine both the power of BERT Representations with Topic Representation, TopicPropagation implementation can be easily implemented:

```python 
predictions_tp = TopicPropagation(model_name = 'all-miniLM-L6-v2', 
                                  docs = docs, 
                                  n_clusters = n_clusters, 
                                  topic_representations = topic_representations)
```

To use the model, refer to `SSBM/ssbm.py`
