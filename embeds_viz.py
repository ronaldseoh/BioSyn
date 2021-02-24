import numpy as np


prev_embeds = np.load('initial.npy')

for i in range(15):
    batch_members = np.load(str(i) + '_topk.npy').flatten()
    batch_embeds = np.load(str(i) + '.npy')
    
    member_changes = []
    nonmember_changes = []

    for j in range(len(batch_embeds)):
        change = np.linalg.norm(batch_embeds[j] - prev_embeds[j])

        if j in batch_members:
            member_changes.append(change)
        else:
            nonmember_changes.append(change)
    
    member_changes_mean = np.mean(member_changes)
    nonmember_changes_mean = np.mean(nonmember_changes)
    
    print(member_changes_mean)
    print(nonmember_changes_mean)
    print()
    
    prev_embeds = batch_embeds
