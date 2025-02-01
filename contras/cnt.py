import os

given_list = [g.split('.json')[0] for g in os.listdir("/mnt/gestalt/home/ddmanddman/beatport_analyze/json")]
# given_list = os.listdir("/mnt/gestalt/home/ddmanddman/beatport_analyze/htdemucs")
path = "/mnt/gestalt/database/beatport/audio/audio"
sel_list = []
print('--- Remaining ---')

problem_list = ["b36be413-daa5-484b-b3cc-78f3a6acfe85.mp3", "c5f8d324-2bdb-4456-b76d-0c12fc8682e1.mp3", 
                "7ac465df-dbee-4e51-9102-6d1c3afb66bc.mp3", "b8a01c12-a84c-4c06-8afa-8158145bacfd.mp3"]

for i in os.listdir(path):
    cnt = 0
    for j in os.listdir(os.path.join(path, i)):
        if j in problem_list:
            print(i, j)
    
    
# for i in os.listdir(path):
#     cnt = 0
#     for j in os.listdir(os.path.join(path, i)):
#         if j.split('.mp3')[0] not in given_list:
#             cnt += 1
#     if cnt != 0 :
#         sel_list.append(i)
    
#     print(i, '---', cnt)