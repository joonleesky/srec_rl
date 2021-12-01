import numpy as np


def compute_recall(ranks, labels, k):
    """
    [Params]
    ranks: np.array(), (batch_size, num_candidates), rank index of each candidate
    labels: np.array(), (batch_size, )
    """
    cut = ranks[:, :k]
    hits = (cut == labels)
    hits = np.sum(hits,1)
    return np.mean(hits)


def compute_ndcg(ranks, labels, k):
    """
    [Params]
    rank: np.array(), (batch_size, num_candidates), rank index of each candidate
    labels: np.array(), (batch_size, )
    """
    cut = ranks[:, :k]
    hits = (cut == labels)
    weights = 1 / np.log2(np.arange(2, 2+k))
    dcg = np.sum((hits * weights), 1)
    idcg = np.ones_like(dcg)
    
    return np.mean(dcg / idcg)


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """
    [Params]
    scores: np.array(), (batch_size, num_candidates), score for each candidate
    labels: np.array(), (batch_size, ), index of the true label (single label)
    ks: [], list of the @k to evaluate
    """
    metrics = {}

    scores = scores
    labels = labels.reshape(-1,1)
    ranks = np.argsort(-scores, 1)
    
    for k in sorted(ks, reverse=True):
        recall_k = compute_recall(ranks, labels, k)
        metrics['Recall@%d' %k] = recall_k
        
        ndcg_k = compute_ndcg(ranks, labels, k)
        metrics['NDCG@%d' %k] = ndcg_k

    return metrics


def average_rewards_precisions_and_recalls_for_ts(rewards, threshold, num_positives, ts):
    """
    [Params]
    rewards: np.array(), (batch_size, timesteps), rating for each interactions
    threshold: int, threshold to divide positive and negative interactions
    num_positives: (batch_size), number of total positive interactions
    ts: [], list of the @t to evaluate
    """
    metrics = {}
    for t in sorted(ts, reverse=True):
        rewards = rewards[:, :t]
        hits = (rewards >= threshold)
        
        rewards_t = np.mean(np.sum(rewards, 1) / t)
        metrics['Avg_Rewards@%d' %t] = rewards_t
        
        precision_t = np.mean(np.sum(hits, 1) / t)
        metrics['Avg_Precision@%d' %t] = precision_t
        
        recall_t = np.sum(hits, 1) / (num_positives + 1e-9)
        recall_t = np.mean(np.clip(recall_t, 0, 1))
        metrics['Avg_Recall@%d' %t] = recall_t
        
    return metrics