import os

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import tqdm


# Plot

# Create a directory for plots
plots_dir = os.path.join("__plots", "tsne")

os.makedirs(plots_dir, exist_ok=True)

color_string_queries = ['red', 'orange', 'green', 'blue', 'purple', 'saddlebrown']
color_string_vocabs = ['salmon', 'moccasin', 'mediumseagreen', 'lightsteelblue', 'violet', 'sandybrown']

for i in tqdm.tqdm(range(722)):
    batch_members = set(np.load(str(i) + '_topk.npy').flatten())
    batch_embeds = np.load(str(i) + '.npy')
    batch_embeds_queries = np.load(str(i) + '_query_embeds.npy')
    batch_topk_by_queries = np.load(str(i) + '_topk_by_queries.npy')
    batch_query_idxs = np.load(str(i) + '_query_idx.npy')
    
    # Apply t-SNE
    #queries_tsne = sklearn.manifold.TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds_queries)
    #vocabs_tsne = sklearn.manifold.TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds)
    
    #np.save(str(i) + '_query_embeds_tsne.npy', queries_tsne)
    #np.save(str(i) + '_tsne.npy', vocabs_tsne)

    # Plot
    plt.figure()
    # plt.xlabel('steps')
    # plt.ylabel('embedding changes (L2 distance)')

    #plt.plot(vocabs_tsne, label='vocabs', color='lightsteelblue')
    #plt.plot(queries_tsne, label='queries', color='moccasin')
    
    # Select random queries and their top-k vocab neighbors
    random_queries = np.random.choice(len(batch_embeds_queries), size=5, replace=False)
    
    neighbors_of_random_queries = [batch_topk_by_queries[q] for q in random_queries]
    
    random_queries_tsne = sklearn.manifold.TSNE(n_components=2, n_jobs=-1).fit_transform(batch_embeds_queries[random_queries])

    neighbors_tsne = []

    for n_list in neighbors_of_random_queries:

        vocab_embeds = [batch_embeds[v] for v in n_list[:10]] # top 5 neighbors
        
        vocab_embeds_tsne = sklearn.manifold.TSNE(n_components=2, n_jobs=-1).fit_transform(vocab_embeds)
        
        neighbors_tsne.append(vocab_embeds_tsne)
        
    for i in range(len(random_queries_tsne)):

        plt.scatter(random_queries_tsne[i], color=color_string_queries[i])
        plt.scatter(neighbors_tsne[i], color=color_string_vocabs[i])

    #plt.legend()
    plt.savefig(os.path.join(plots_dir, str(i) + '_tsne.png'))
    plt.close()
