import pandas as pd
import numpy as np
dict1={}
species="Arabidopsis thaliana"
#species="Oryza Sativa"
overlap=True
req_no=0
mask_no=1
unit_number_temp=[10,20]
unit_number=unit_number_temp[req_no]
masked=["non_masked","masked"]
req_mask=masked[mask_no]
df1=pd.read_csv(f"/s/chromatin/b/nobackup/deepplant/Data/genomes_chunked_{unit_number}kb_{req_mask}.csv")
df1=df1[df1["genome"]==species].reset_index()
chrom=[]
index=[]
dict1={}
list_conn=[]
list_keys=[]
if species=="Arabidopsis thaliana":
    if overlap:
        extend_cal_temp=[[0,3043,5013,7359,9218],[0,1522,2508,3681,4611]]
    else:
        extend_cal_temp=[[0,3043,5013,7359,9218],[0,1522,2508,3681,4611]]
    species_no="arabidopsis_thaliana"
    no_chromosomes=6
else:
    if overlap:
        extend_cal_temp=[[11916,16244,19839, 23481,27032,30029, 33154,36124,38970,41272,43593,46496],[5961,8126,9924,11745,13521,15020, 16583, 18068, 19492, 20644, 21805, 23257]]

    else:
        extend_cal_temp=[[11916,16244,19839, 23481,27032,30029, 33154,36124,38970,41272,43593,46496],[5961,8126,9924,11745,13521,15020, 16583, 18068, 19492, 20644, 21805, 23257]]
    species_no="Oryza_Sativa"
    no_chromosomes=13
import pickle as pkl
for i in range(1,no_chromosomes):
    chrom.append(df1[df1["chromosome"]==f"chromosome {i}"].reset_index())
    index.append(chrom[-1]["seq_idx"])
    for j in chrom[-1]["seq_idx"]:
        dict1[j]=[]
        list_keys.append(j)
    
extend_cal=extend_cal_temp[req_no]
#extend_cal=[0,6086,5012,10025,17595,21312]
#extend_cal=[0,1522,2508,4401,5331]
#extend_cal=[0,10143,16707,29323,35518]
size=int(unit_number)*1000
for i in range(1,no_chromosomes):
    filename=f"Result_seq_chrom_{req_mask}_{species_no}_{unit_number}kb_{i}.csv"
    df2=pd.read_csv(filename)
    df2=df2[df2["similarity"]>70].reset_index()
    df2=df2[df2["width"]>=1000].reset_index()
    for j in range(len(df2)):
        rstart=df2["sstart"][j]
        chr_no=int(df2["chr2"][j][-1])
        rend=df2["send"][j]
        quo=int(rstart/size)
        rem=rend%size
        quo=quo+extend_cal[chr_no-1]
        #print(quo,rem,rstart,rend)
        if rem!=0:
            quo=quo+1
        #if req_seq_index not in list(dict1.keys()):
        #req_seq_index1=list(dict1.keys())[-1]
        #print(f"----------->{req_seq_index1}") 
        if quo>len(list_keys)-1:
             quo=len(list_keys)-1
        req_seq_index=f"seq_{quo}"
        try:
            if req_seq_index not in dict1[df2["seq_idx"][j]] and req_seq_index!=df2["seq_idx"][j]:    
                dict1[df2["seq_idx"][j]].append(req_seq_index)
            if df2["seq_idx"][j] not in dict1[req_seq_index] and req_seq_index!=df2["seq_idx"][j]:
                dict1[req_seq_index].append(df2["seq_idx"][j])
        except:
            pass
#pkl.save("neighbours.pkl",dict1)

with open(f'neighbours_{unit_number}kb_1000_70_{req_mask}_{species_no}.pkl', 'wb') as handle:
    pkl.dump(dict1, handle, protocol=pkl.HIGHEST_PROTOCOL)

def iterative_dfs(graph, start):
    """
    Perform an iterative DFS from the start node.
    """
    visited = set()
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(neighbour for neighbour in graph.get(node, []) if neighbour not in visited)
    return visited

def find_connected_components(graph,visited):
    """
    Find all connected components in the graph using iterative DFS.
    """
    components = []
    
    for node in graph:
        new_comp=[]
        if node not in visited:
            #visited.append(node)
            #new_comp.append(node) 
            component = iterative_dfs(graph, node)
            for p in component:
                if p not in  visited:
                    #print(f'{p}-----{visited}')
                    visited.append(p)
                    new_comp.append(p)
            components.append(new_comp)
    
    return components




visited = []
components=find_connected_components(dict1,visited)
print(len(components))
for k in range(len(components)):
    if len(components[k])>0:
        print(f'{k}---->{len(components[k])}')
        for k1 in components[k]:
            print(k1)

