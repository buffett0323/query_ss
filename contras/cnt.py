import os

given_list = [g.split('.json')[0] for g in os.listdir("/mnt/gestalt/home/ddmanddman/beatport_analyze/json")]
path = "/mnt/gestalt/database/beatport/audio/audio"
sel_list = []
print('--- Remaining ---')

for i in os.listdir(path):
    cnt = 0
    for j in os.listdir(os.path.join(path, i)):
        if j.split('.mp3')[0] not in given_list:
            cnt += 1
    if cnt != 0 :
        sel_list.append(i)
    
    print(i, '---', cnt)