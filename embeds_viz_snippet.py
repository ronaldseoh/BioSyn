import numpy as np
import matplotlib.pyplot as plt


prev_embeds = np.load('initial.npy')
prev_topk_by_queries = np.load('initial_topk_by_queries.npy')

member_embeds_avg_change_history = []
nonmember_embeds_avg_change_history = []

member_query_num_neighbors_swapped_avg_history = []
member_query_num_neighbors_swapped_min_history = []
member_query_num_neighbors_swapped_max_history = []

nonmember_query_num_neighbors_swapped_avg_history = []
nonmember_query_num_neighbors_swapped_min_history = []
nonmember_query_num_neighbors_swapped_max_history = []

member_query_neighbors_zero_change_count_history = []
nonmember_query_neighbors_zero_change_count_history = []

for i in range(50):
    batch_members = set(np.load(str(i) + '_topk.npy').flatten())
    batch_embeds = np.load(str(i) + '.npy')
    batch_topk_by_queries = np.load(str(i) + '_topk_by_queries.npy')
    batch_query_idxs = np.load(str(i) + '_query_idx.npy')
    
    member_embeds_changes = []
    nonmember_embeds_changes = []
    
    changes = np.linalg.norm(batch_embeds - prev_embeds, ord=2, axis=1)

    for j in range(len(changes)):
        if j in batch_members:
            member_embeds_changes.append(changes[j])
        else:
            nonmember_embeds_changes.append(changes[j])
    
    member_embeds_changes_avg = np.mean(member_embeds_changes)
    nonmember_embeds_changes_avg = np.mean(nonmember_embeds_changes)
    
    member_embeds_avg_change_history.append(member_embeds_changes_avg)
    nonmember_embeds_avg_change_history.append(nonmember_embeds_changes_avg)
    
    member_query_num_neighbors_swapped = []
    nonmember_query_num_neighbors_swapped = []
    
    member_query_zero_neighbors_change_count = 0
    nonmember_query_zero_neighbors_change_count = 0
    
    for j in range(len(batch_topk_by_queries)):
        nn_prev = set(prev_topk_by_queries[j])
        nn_cur = set(batch_topk_by_queries[j])
        
        num_swapped = len(nn_prev - nn_cur)
        
        if j in batch_query_idxs:
            member_query_num_neighbors_swapped.append(num_swapped)
        else:
            nonmember_query_num_neighbors_swapped.append(num_swapped)
        
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
    
    print("Member embedding avg change =", member_embeds_changes_avg)
    print("Non-member embedding avg change =", nonmember_embeds_changes_avg)

    print("Avg. # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_avg)
    print("Min # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_min)
    print("Max # of neighbors swapped per query in the batch =", member_query_num_neighbors_swapped_max)
    
    print("Avg. # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_avg)
    print("Min # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_min)
    print("Max # of neighbors swapped per query NOT in the batch =", nonmember_query_num_neighbors_swapped_max)

    print("# of zero neighbor changes for query in the batch =", member_query_zero_neighbors_change_count, "out of", len(batch_query_idxs))
    print("# of zero neighbor changes for query NOT in the batch =", nonmember_query_zero_neighbors_change_count, "out of", len(batch_topk_by_queries) - len(batch_query_idxs))
    
    print()
    
    prev_embeds = batch_embeds
    prev_topk_by_queries = batch_topk_by_queries

# Plot

# Average embeddings of dense representations in the dictionary
plt.figure()
plt.xlabel('steps')
plt.ylabel('Average embedding changes')

plt.plot(
    list(range(len(member_embeds_avg_change_history))),
    member_embeds_avg_change_history,
    label='within batch')
    
plt.plot(
    list(range(len(nonmember_embeds_avg_change_history))),
    nonmember_embeds_avg_change_history,
    label='outside batch')

plt.legend()
plt.savefig('embeds_avg_change_history.png')
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
plt.savefig('member_query_num_neighbors_swapped_history.png')
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
plt.savefig('nonmember_query_num_neighbors_swapped_history.png')
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
plt.savefig('zero_neighbors_change_count_history.png')
plt.close()
