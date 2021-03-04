import os

import numpy as np
import matplotlib.pyplot as plt


prev_embeds = np.load('initial.npy')
prev_embeds_queries = np.load('initial_query_embeds.npy')
prev_topk_by_queries = np.load('initial_topk_by_queries.npy')

member_embeds_avg_change_history = []
member_embeds_min_change_history = []
member_embeds_max_change_history = []

nonmember_embeds_avg_change_history = []
nonmember_embeds_min_change_history = []
nonmember_embeds_max_change_history = []

member_query_num_neighbors_swapped_avg_history = []
member_query_num_neighbors_swapped_min_history = []
member_query_num_neighbors_swapped_max_history = []

nonmember_query_num_neighbors_swapped_avg_history = []
nonmember_query_num_neighbors_swapped_min_history = []
nonmember_query_num_neighbors_swapped_max_history = []

member_query_neighbors_zero_change_count_history = []
nonmember_query_neighbors_zero_change_count_history = []

member_query_nearest_distance_avg_history = []
member_query_nearest_distance_min_history = []
member_query_nearest_distance_max_history = []

member_query_embed_change_avg_history = []
member_query_embed_change_min_history = []
member_query_embed_change_max_history = []

nonmember_query_nearest_distance_avg_history = []
nonmember_query_nearest_distance_min_history = []
nonmember_query_nearest_distance_max_history = []

nonmember_query_embed_change_avg_history = []
nonmember_query_embed_change_min_history = []
nonmember_query_embed_change_max_history = []

for i in range(60):
    batch_members = set(np.load(str(i) + '_topk.npy').flatten())
    batch_embeds = np.load(str(i) + '.npy')
    batch_embeds_queries = np.load(str(i) + '_query_embeds.npy')
    batch_topk_by_queries = np.load(str(i) + '_topk_by_queries.npy')
    batch_query_idxs = np.load(str(i) + '_query_idx.npy')
    
    member_embeds_changes = []
    nonmember_embeds_changes = []
    
    changes = np.linalg.norm(batch_embeds - prev_embeds, ord=2, axis=1)

    for k in range(len(changes)):
        if k in batch_members:
            member_embeds_changes.append(changes[k])
        else:
            nonmember_embeds_changes.append(changes[k])
    
    member_embeds_changes_avg = np.mean(member_embeds_changes)
    member_embeds_changes_min = np.min(member_embeds_changes)
    member_embeds_changes_max = np.max(member_embeds_changes)
    
    member_embeds_avg_change_history.append(member_embeds_changes_avg)
    member_embeds_min_change_history.append(member_embeds_changes_min)
    member_embeds_max_change_history.append(member_embeds_changes_max)

    nonmember_embeds_changes_avg = np.mean(nonmember_embeds_changes)
    nonmember_embeds_changes_min = np.min(nonmember_embeds_changes)
    nonmember_embeds_changes_max = np.max(nonmember_embeds_changes)
    
    nonmember_embeds_avg_change_history.append(nonmember_embeds_changes_avg)
    nonmember_embeds_min_change_history.append(nonmember_embeds_changes_min)
    nonmember_embeds_max_change_history.append(nonmember_embeds_changes_max)
    
    member_query_num_neighbors_swapped = []
    nonmember_query_num_neighbors_swapped = []
    
    member_query_zero_neighbors_change_count = 0
    nonmember_query_zero_neighbors_change_count = 0
    
    member_query_nearest_distances = []
    member_query_embed_changes = []

    nonmember_query_nearest_distances = []
    nonmember_query_embed_changes = []
    
    for j in range(len(batch_topk_by_queries)):
        nn_prev = set(prev_topk_by_queries[j])
        nn_cur = set(batch_topk_by_queries[j])
        
        num_swapped = len(nn_prev - nn_cur)
        
        nearest_point_embed = batch_embeds[batch_topk_by_queries[j][0]]
        
        query_embed_cur = batch_embeds_queries[j]
        query_embed_prev = prev_embeds_queries[j]
        
        distance_to_nearest = np.linalg.norm(
            query_embed_cur - nearest_point_embed, ord=2)
            
        query_embed_changes = np.linalg.norm(
            query_embed_cur - query_embed_prev, ord=2)

        if j in batch_query_idxs:
            member_query_num_neighbors_swapped.append(num_swapped)
            
            member_query_nearest_distances.append(distance_to_nearest)
            member_query_embed_changes.append(query_embed_changes)
        else:
            nonmember_query_num_neighbors_swapped.append(num_swapped)
            
            nonmember_query_nearest_distances.append(distance_to_nearest)
            nonmember_query_embed_changes.append(query_embed_changes)
        
        if num_swapped == 0:
            if j in batch_query_idxs:
                member_query_zero_neighbors_change_count = member_query_zero_neighbors_change_count + 1
            else:
                nonmember_query_zero_neighbors_change_count = nonmember_query_zero_neighbors_change_count + 1
        
    member_query_num_neighbors_swapped_avg = np.mean(member_query_num_neighbors_swapped)
    member_query_num_neighbors_swapped_avg_history.append(member_query_num_neighbors_swapped_avg)
    
    member_query_num_neighbors_swapped_min = np.min(member_query_num_neighbors_swapped)
    member_query_num_neighbors_swapped_min_history.append(member_query_num_neighbors_swapped_min)
    
    member_query_num_neighbors_swapped_max = np.max(member_query_num_neighbors_swapped)
    member_query_num_neighbors_swapped_max_history.append(member_query_num_neighbors_swapped_max)
    
    nonmember_query_num_neighbors_swapped_avg = np.mean(nonmember_query_num_neighbors_swapped)
    nonmember_query_num_neighbors_swapped_avg_history.append(nonmember_query_num_neighbors_swapped_avg)
    
    nonmember_query_num_neighbors_swapped_min = np.min(nonmember_query_num_neighbors_swapped)
    nonmember_query_num_neighbors_swapped_min_history.append(nonmember_query_num_neighbors_swapped_min)
    
    nonmember_query_num_neighbors_swapped_max = np.max(nonmember_query_num_neighbors_swapped)
    nonmember_query_num_neighbors_swapped_max_history.append(nonmember_query_num_neighbors_swapped_max)

    member_query_neighbors_zero_change_count_history.append(member_query_zero_neighbors_change_count / len(batch_query_idxs))
    nonmember_query_neighbors_zero_change_count_history.append(nonmember_query_zero_neighbors_change_count / (len(batch_topk_by_queries) - len(batch_query_idxs)))
    
    member_query_nearest_distance_avg = np.mean(member_query_nearest_distances)
    member_query_nearest_distance_avg_history.append(member_query_nearest_distance_avg)
    
    member_query_nearest_distance_min = np.min(member_query_nearest_distances)
    member_query_nearest_distance_min_history.append(member_query_nearest_distance_min)
    
    member_query_nearest_distance_max = np.max(member_query_nearest_distances)
    member_query_nearest_distance_max_history.append(member_query_nearest_distance_max)
    
    member_query_embed_change_avg = np.mean(member_query_embed_changes)
    member_query_embed_change_avg_history.append(member_query_embed_change_avg)
    
    member_query_embed_change_min = np.min(member_query_embed_changes)
    member_query_embed_change_min_history.append(member_query_embed_change_min)
    
    member_query_embed_change_max = np.max(member_query_embed_changes)
    member_query_embed_change_max_history.append(member_query_embed_change_max)
    
    nonmember_query_nearest_distance_avg = np.mean(nonmember_query_nearest_distances)
    nonmember_query_nearest_distance_avg_history.append(nonmember_query_nearest_distance_avg)
    
    nonmember_query_nearest_distance_min = np.min(nonmember_query_nearest_distances)
    nonmember_query_nearest_distance_min_history.append(nonmember_query_nearest_distance_min)
    
    nonmember_query_nearest_distance_max = np.max(nonmember_query_nearest_distances)
    nonmember_query_nearest_distance_max_history.append(nonmember_query_nearest_distance_max)
    
    nonmember_query_embed_change_avg = np.mean(nonmember_query_embed_changes)
    nonmember_query_embed_change_avg_history.append(nonmember_query_embed_change_avg)
    
    nonmember_query_embed_change_min = np.min(nonmember_query_embed_changes)
    nonmember_query_embed_change_min_history.append(nonmember_query_embed_change_min)
    
    nonmember_query_embed_change_max = np.max(nonmember_query_embed_changes)
    nonmember_query_embed_change_max_history.append(nonmember_query_embed_change_max)
    

    print("Batch %d:" % i)
    print()
    
    print("Member embedding avg change =", member_embeds_changes_avg)
    print("Member embedding min change =", member_embeds_changes_min)
    print("Member embedding max change =", member_embeds_changes_max)
    print()
    
    print("Non-member embedding avg change =", nonmember_embeds_changes_avg)
    print("Non-member embedding min change =", nonmember_embeds_changes_min)
    print("Non-member embedding max change =", nonmember_embeds_changes_max)
    print()

    print("Avg. # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_avg)
    print("Min # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_min)
    print("Max # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_max)
    print()
    
    print("Avg. # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_avg)
    print("Min # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_min)
    print("Max # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_max)
    print()

    print("# of zero neighbor changes for query in the batch =", member_query_zero_neighbors_change_count, "out of", len(batch_query_idxs))
    print("# of zero neighbor changes for query NOT in the batch =", nonmember_query_zero_neighbors_change_count, "out of", len(batch_topk_by_queries) - len(batch_query_idxs))
    print()
    
    print("Avg. nearest distance per query in the batch =", member_query_nearest_distance_avg)
    print("Min nearest distance per query in the batch =", member_query_nearest_distance_min)
    print("Max nearest distance per query in the batch =", member_query_nearest_distance_max)
    print()
    
    print("Avg. nearest distance per query NOT in the batch =", nonmember_query_nearest_distance_avg)
    print("Min nearest distance per query NOT in the batch =", nonmember_query_nearest_distance_min)
    print("Max nearest distance per query NOT in the batch =", nonmember_query_nearest_distance_max)
    print()

    print("Avg. query embed changes in the batch =", member_query_embed_change_avg)
    print("Min query embed changes in the batch =", member_query_embed_change_min)
    print("Max query embed changes in the batch =", member_query_embed_change_max)
    print()
    
    print("Avg. query embed changes NOT in the batch =", nonmember_query_embed_change_avg)
    print("Min query embed changes NOT in the batch =", nonmember_query_embed_change_min)
    print("Max query embed changes NOT in the batch =", nonmember_query_embed_change_max)
    print()
        
    print()
    
    prev_embeds = batch_embeds
    prev_topk_by_queries = batch_topk_by_queries
    prev_embeds_queries = batch_embeds_queries

# Plot

# Create a directory for plots
plots_dir = os.path.join("__plots")

os.makedirs(plots_dir, exist_ok=True)

# Average embedding changes of dense representations in the dictionary
plt.figure()
plt.xlabel('steps')
plt.ylabel('Average embedding changes (L2 distance)')

plt.plot(
    list(range(len(member_embeds_avg_change_history))),
    member_embeds_avg_change_history,
    label='within batch')
    
plt.plot(
    list(range(len(nonmember_embeds_avg_change_history))),
    nonmember_embeds_avg_change_history,
    label='outside batch')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'embeds_avg_change_history.png'))
plt.close()

# Stats for embeddings changes of dense representations in the dictionary within batch
plt.figure()
plt.xlabel('steps')
plt.ylabel('embedding changes (L2 distance)')

plt.plot(
    list(range(len(member_embeds_avg_change_history))),
    member_embeds_avg_change_history,
    label='avg')
    
plt.plot(
    list(range(len(member_embeds_max_change_history))),
    member_embeds_max_change_history,
    label='max')

plt.plot(
    list(range(len(member_embeds_min_change_history))),
    member_embeds_min_change_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'member_embeds_change_history.png'))
plt.close()

# Stats for embeddings changes of dense representations in the dictionary OUTSIDE batch
plt.figure()
plt.xlabel('steps')
plt.ylabel('embedding changes (L2 distance)')

plt.plot(
    list(range(len(nonmember_embeds_avg_change_history))),
    nonmember_embeds_avg_change_history,
    label='avg')
    
plt.plot(
    list(range(len(nonmember_embeds_max_change_history))),
    nonmember_embeds_max_change_history,
    label='max')

plt.plot(
    list(range(len(nonmember_embeds_min_change_history))),
    nonmember_embeds_min_change_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nonmember_embeds_change_history.png'))
plt.close()

# Number of neighbors swapped per query (within batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('# neighbors swapped (per query)')

plt.plot(
    list(range(len(member_query_num_neighbors_swapped_avg_history))),
    member_query_num_neighbors_swapped_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(member_query_num_neighbors_swapped_max_history))),
    member_query_num_neighbors_swapped_max_history,
    label='max')

plt.plot(
    list(range(len(member_query_num_neighbors_swapped_min_history))),
    member_query_num_neighbors_swapped_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'member_query_num_neighbors_swapped_history.png'))
plt.close()

# Number of neighbors swapped per query (OUTSIDE batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('# neighbors swapped (per query)')

plt.plot(
    list(range(len(nonmember_query_num_neighbors_swapped_avg_history))),
    nonmember_query_num_neighbors_swapped_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(nonmember_query_num_neighbors_swapped_max_history))),
    nonmember_query_num_neighbors_swapped_max_history,
    label='max')

plt.plot(
    list(range(len(nonmember_query_num_neighbors_swapped_min_history))),
    nonmember_query_num_neighbors_swapped_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nonmember_query_num_neighbors_swapped_history.png'))
plt.close()

# Proportion of queries with ZERO neighbor changes (in/out batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('proportion of queries w/ ZERO neighbor changes')

plt.plot(
    list(range(len(member_query_neighbors_zero_change_count_history))),
    member_query_neighbors_zero_change_count_history,
    label='queries in batch')
    
plt.plot(
    list(range(len(nonmember_query_neighbors_zero_change_count_history))),
    nonmember_query_neighbors_zero_change_count_history,
    label='queries NOT in batch')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'zero_neighbors_change_count_history.png'))
plt.close()

# Nearest distance per query (within batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('distance to the nearest neighbor')

plt.plot(
    list(range(len(member_query_nearest_distance_avg_history))),
    member_query_nearest_distance_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(member_query_nearest_distance_max_history))),
    member_query_nearest_distance_max_history,
    label='max')

plt.plot(
    list(range(len(member_query_nearest_distance_min_history))),
    member_query_nearest_distance_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'member_query_nearest_distance_history.png'))
plt.close()

# Nearest distance per query (outside batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('distance to the nearest neighbor')

plt.plot(
    list(range(len(nonmember_query_nearest_distance_avg_history))),
    nonmember_query_nearest_distance_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(nonmember_query_nearest_distance_max_history))),
    nonmember_query_nearest_distance_max_history,
    label='max')

plt.plot(
    list(range(len(nonmember_query_nearest_distance_min_history))),
    nonmember_query_nearest_distance_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nonmember_query_nearest_distance_history.png'))
plt.close()

# query embed changes per query (within batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('query embedding changes (L2 distance)')

plt.plot(
    list(range(len(member_query_embed_change_avg_history))),
    member_query_embed_change_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(member_query_embed_change_max_history))),
    member_query_embed_change_max_history,
    label='max')

plt.plot(
    list(range(len(member_query_embed_change_min_history))),
    member_query_embed_change_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'member_query_embed_change_history.png'))
plt.close()

# query embed changes per query (outside batch)
plt.figure()
plt.xlabel('steps')
plt.ylabel('query embedding changes (L2 distance)')

plt.plot(
    list(range(len(nonmember_query_embed_change_avg_history))),
    nonmember_query_embed_change_avg_history,
    label='avg')
    
plt.plot(
    list(range(len(nonmember_query_embed_change_max_history))),
    nonmember_query_embed_change_max_history,
    label='max')

plt.plot(
    list(range(len(nonmember_query_embed_change_min_history))),
    nonmember_query_embed_change_min_history,
    label='min')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nonmember_query_embed_change_history.png'))
plt.close()

# avg distance change + avg query embed change
plt.figure()
plt.xlabel('avg distance to the nearest neighbor')
plt.ylabel('avg query embed change')

plt.scatter(
    member_query_nearest_distance_avg_history,
    member_query_embed_change_avg_history,
    label='queries in batch')
    
plt.scatter(
    nonmember_query_nearest_distance_avg_history,
    nonmember_query_embed_change_avg_history,
    label='queries NOT in batch')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nearest_distance_query_embed_change_history.png'))
plt.close()

# avg distance change + number of neighbors changed
plt.figure()
plt.xlabel('avg distance to the nearest neighbor')
plt.ylabel('avg number of neighbors changed')

plt.scatter(
    member_query_nearest_distance_avg_history,
    member_query_num_neighbors_swapped_avg_history,
    label='queries in batch')
    
plt.scatter(
    nonmember_query_nearest_distance_avg_history,
    nonmember_query_num_neighbors_swapped_avg_history,
    label='queries NOT in batch')

plt.legend()
plt.savefig(os.path.join(plots_dir, 'nearest_distance_num_neighbors_swapped_history.png'))
plt.close()
