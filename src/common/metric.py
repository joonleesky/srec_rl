import torch


def recall(ranks, labels, k):
    """
    [Params]
    ranks: (batch_size, num_candidates), rank index of each candidate
    labels: (batch_size, )
    """
    cut = ranks[:, :k]
    hits = (cut == labels)
    hits = torch.sum(hits,1).float()
    return torch.mean(hits).item()


def ndcg(ranks, labels, k):
    """
    [Params]
    rank: (batch_size, num_candidates), rank index of each candidate
    labels: (batch_size, )
    """
    cut = ranks[:, :k]
    hits = (cut == labels)
    weights = 1 / torch.log2(torch.arange(2, 2+k).float())
    dcg = torch.sum((hits.float() * weights), 1).float()
    idcg = torch.ones_like(dcg)
    
    return (dcg / idcg).mean().item()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """
    [Params]
    scores: (batch_size, num_candidates), score for each candidate
    labels: (batch_size, ), index of the true label (single label)
    ks: [], list of the @k to evaluate
    """
    metrics = {}

    scores = scores.cpu()
    labels = labels.view(-1,1).cpu()
    ranks = (-scores).argsort(dim=1)
    
    for k in sorted(ks, reverse=True):
        recall_k = recall(ranks, labels, k)
        metrics['Recall@%d' %k] = recall_k
        
        ndcg_k = ndcg(ranks, labels, k)
        metrics['NDCG@%d' %k] = ndcg_k

    return metrics