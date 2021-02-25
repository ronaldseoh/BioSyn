import numpy as np
import matplotlib.pyplot as plt


prev_embeds = np.load('initial.npy')
prev_topk_by_queries = np.load('initial_topk_by_queries.npy')

member_embeds_avg_change_history = []
nonmember_embeds_avg_change_history = []

num_neighbors_swapped_avg_history = []
num_neighbors_swapped_min_history = []
num_neighbors_swapped_max_history = []
no_neighbors_change_count_history = []

for i in range(150):
    batch_members = set(np.load(str(i) + '_topk.npy').flatten())
    batch_embeds = np.load(str(i) + '.npy')
    batch_topk_by_queries = np.load(str(i) + '_topk_by_queries.npy')
    
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
    
    num_neighbors_swapped_by_queries = []
    
    no_neighbors_change_count = 0
    
    for j in range(len(batch_topk_by_queries)):
        nn_prev = set(prev_topk_by_queries[j])
        nn_cur = set(batch_topk_by_queries[j])
        
        num_swapped = len(nn_prev - nn_cur)
        
        num_neighbors_swapped_by_queries.append(num_swapped)
        
        if num_swapped == 0:
            no_neighbors_change_count = no_neighbors_change_count + 1
        
    num_neighbors_swapped_avg = np.mean(num_neighbors_swapped_by_queries)
    num_neighbors_swapped_avg_history.append(num_neighbors_swapped_avg)
    
    num_neighbors_swapped_min = np.min(num_neighbors_swapped_by_queries)
    num_neighbors_swapped_min_history.append(num_neighbors_swapped_min)
    
    num_neighbors_swapped_max = np.max(num_neighbors_swapped_by_queries)
    num_neighbors_swapped_max_history.append(num_neighbors_swapped_max)

    no_neighbors_change_count_history.append(no_neighbors_change_count)
    
    print("Member embedding avg change =", member_embeds_changes_avg)
    print("Non-member embedding avg change =", nonmember_embeds_changes_avg)
    print("Avg. # of neighbors swapped per query =", num_neighbors_swapped_avg)
    print("Min # of neighbors swapped per query =", num_neighbors_swapped_min)
    print("Max # of neighbors swapped per query =", num_neighbors_swapped_max)
    print("# of no neighbor changes=", no_neighbors_change_count, "out of", len(batch_topk_by_queries))
    
    print()
    
    prev_embeds = batch_embeds
    prev_topk_by_queries = batch_topk_by_queries

np.save('member_embeds_avg_change_history.npy', member_embeds_avg_change_history)
np.save('nonmember_embeds_avg_change_history.npy', nonmember_embeds_avg_change_history)
np.save('num_neighbors_swapped_avg_history.npy', num_neighbors_swapped_avg_history)

# Plot

plt.figure()
plt.xlabel('steps')
plt.ylabel('Average embedding changes (within batch)')

plt.plot(list(range(len(member_embeds_avg_change_history))), member_embeds_avg_change_history)

plt.savefig('member_embeds_avg_change_history.png')
plt.close()

plt.figure()
plt.xlabel('steps')
plt.ylabel('Average embedding changes (outside batch)')

plt.plot(list(range(len(nonmember_embeds_avg_change_history))), nonmember_embeds_avg_change_history)

plt.savefig('nonmember_embeds_avg_change_history.png')
plt.close()

plt.figure()
plt.xlabel('steps')
plt.ylabel('avg # neighbors swapped (per query)')

plt.plot(list(range(len(num_neighbors_swapped_avg_history))), num_neighbors_swapped_avg_history)

plt.savefig('num_neighbors_swapped_avg_history.png')
plt.close()

plt.figure()
plt.xlabel('steps')
plt.ylabel('min # neighbors swapped (per query)')

plt.plot(list(range(len(num_neighbors_swapped_min_history))), num_neighbors_swapped_min_history)

plt.savefig('num_neighbors_swapped_min_history.png')
plt.close()

plt.figure()
plt.xlabel('steps')
plt.ylabel('max # neighbors swapped (per query)')

plt.plot(list(range(len(num_neighbors_swapped_max_history))), num_neighbors_swapped_max_history)

plt.savefig('num_neighbors_swapped_max_history.png')
plt.close()

plt.figure()
plt.xlabel('steps')
plt.ylabel('# of no neighbor changes')

plt.plot(list(range(len(no_neighbors_change_count_history))), no_neighbors_change_count_history)

plt.savefig('no_neighbors_change_count_history.png')
plt.close()
