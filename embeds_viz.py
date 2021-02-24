import numpy as np


prev_embeds = np.load('initial.npy')
member_stats = []
nonmember_stats = []

for i in range(150):
    batch_members = set(np.load(str(i) + '_topk.npy').flatten())
    batch_embeds = np.load(str(i) + '.npy')
    
    member_changes = []
    nonmember_changes = []
    
    changes = np.linalg.norm(batch_embeds - prev_embeds, ord=2, axis=1)

    for j in range(len(changes)):
        if j in batch_members:
            member_changes.append(changes[j])
        else:
            nonmember_changes.append(changes[j])
    
    member_changes_mean = np.mean(member_changes)
    nonmember_changes_mean = np.mean(nonmember_changes)
    
    member_stats.append(member_changes_mean)
    nonmember_stats.append(nonmember_changes_mean)
    
    print(member_changes_mean)
    print(nonmember_changes_mean)
    print()
    
    prev_embeds = batch_embeds

np.save('member_stats.npy', member_stats)
np.save('nonmember_stats.npy', nonmember_stats)
