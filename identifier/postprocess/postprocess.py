import os
import numpy as np
# result_path = "../output.npz"
# input_path = "../tracklets.txt"
# output_path = "../track3.txt"
# result = np.load(result_path)
# dis_thre = 12
# dis_remove = 100
# max_length = 20

def calc_reid(result,dis_remove=100,dis_thre=12):
    distmat=result["distmat"]
    q_pids = result["q_pids"]
    g_pids = result["g_pids"]
    q_camids=result["q_camids"]+1
    g_camids=result["g_camids"]+1
    new_id = np.max(g_pids)
    # print(np.max(g_pids))
    # print(np.max(q_pids))
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(distmat, axis=1)
    num_q, num_g = distmat.shape
    # print(np.min(distmat))
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_thre)
        keep = np.invert(remove)

        remove_hard = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (distmat[q_idx][order]>dis_remove)
        keep_hard = np.invert(remove_hard)
        if True not in keep_hard:
            if q_camid not in list(rm_dict.keys()):
                rm_dict[q_camid] = {}
            rm_dict[q_camid][q_pid] = True
        sel_g_dis = distmat[q_idx][order][keep]
        sel_g_pids = g_pids[order][keep]
        sel_g_camids = g_camids[order][keep]
        sel_g_pids_list = []
        sel_g_camids_list = []
        selg_dis_list = []
        for i in range(sel_g_pids.shape[0]):
            sel_pid =  sel_g_pids[i]
            sel_cam = sel_g_camids[i]
            sel_dis = sel_g_dis[i]
            if sel_cam not in sel_g_camids_list and sel_cam!=q_camid:
                sel_g_pids_list.append(sel_pid)
                sel_g_camids_list.append(sel_cam)
                selg_dis_list.append(sel_dis)
                

        if len(selg_dis_list)>0:
            new_id+=1
            if q_camid in list(reid_dict.keys()):
                if q_pid in list(reid_dict[q_camid]):
                    if reid_dict[q_camid][q_pid]["dis"]>min(selg_dis_list):
                        reid_dict[q_camid][q_pid]["dis"] = min(selg_dis_list)
                        reid_dict[q_camid][q_pid]["id"] = new_id
                else:
                    reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}
            else:
                reid_dict[q_camid] = {}
                reid_dict[q_camid][q_pid] = {"dis":min(selg_dis_list),"id":new_id}


        for i in range(len(sel_g_pids_list)):
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_pids_list[i] in list(reid_dict[sel_g_camids_list[i]]):
                    if reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"]>selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["id"] = new_id
                else:
                    reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {"dis":selg_dis_list[i],"id":new_id}
    
    return reid_dict,rm_dict
                
def calc_length(output):
    calc_dict = {}
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id not in list(calc_dict.keys()):
            calc_dict[cam_id] = {}
        if track_id not in list(calc_dict[cam_id].keys()):
            calc_dict[cam_id][track_id] = 1
        else:
            calc_dict[cam_id][track_id]+=1
    return calc_dict


def update_output(output,reid_dict,rm_dict,f,max_length=20):
    calc_dict = calc_length(output)
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id in list(rm_dict.keys()):
            if track_id in list(rm_dict[cam_id].keys()):
                continue
        if calc_dict[cam_id][track_id] < max_length:
            continue
        if cam_id in list(reid_dict.keys()):
            if track_id in list(reid_dict[cam_id].keys()):
                line[1] = str(reid_dict[cam_id][track_id]["id"])
        f.write(" ".join(line)+"\n")






# if __name__ == "__main__":
#     reid_dict,rm_dict = calc_reid(result)
#     print(rm_dict,reid_dict)
#     with open(input_path,"r") as f:
#         or_tracks = f.readlines()
#     g = open(output_path,"w")
#     update_output(or_tracks,reid_dict,rm_dict,g)



