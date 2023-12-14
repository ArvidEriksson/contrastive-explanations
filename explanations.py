def mean_contrast(explanation_vectors, target):
    # Shape of explanation vectors should be (C, ...) where C is the number of classes
    # Each explanation vector contains the explanation for each pixel.
    target_explanation = explanation_vectors[target]
    mask = np.arange(explanation_vectors.shape[0]) != target
    mean_explanation_excluding_target = explanation_vectors[mask].mean(axis=0)
    explanation = target_explanation - mean_explanation_excluding_target
    
    return min_max_normalize(explanation)

def max_contrast(explanation_vectors, target, logits):
    # Shape of explanation vectors should be (C, ...) where C is the number of classes
    # Each explanation vector contains the explanation for each pixel.
    _, inds = torch.topk(logits, 2, dim=1)

    inds = inds[0]
    print(inds[1])
    ind = inds[0] if inds[0] != target else inds[1]
    
    explanation = explanation_vectors[target] - explanation_vectors[ind]
    
    return min_max_normalize(explanation)
    
def weighted_contrast(explanation_vectors, target, logits):
    # Shape of explanation vectors should be (C, ...) where C is the number of classes
    # Each explanation vector contains the explanation for each pixel.
    
    # This is very inefficient
    logits_cp = logits.detach().clone()
    logits_cp[0][target] = float("-inf")
    alphas = torch.nn.functional.softmax(logits_cp, dim=1).cpu().numpy()
    #alphas = np.array(alphas)

    explanation = np.zeros_like(explanation_vectors[0])

    # Iterate over each class and its corresponding weight
    for i, (explanation_vector, alpha) in enumerate(zip(explanation_vectors, alphas[0])):
        if i != target:
            explanation -= alpha * explanation_vector
        else:
            explanation += explanation_vector
            
    return min_max_normalize(explanation)
