import cv2
import os
import argparse
import numpy as np


def orb_sim(im1,im2):
    try:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp1 = orb.detect(im1,None)
        kp1, des1 = orb.compute(im1, kp1)
        kp2 = orb.detect(im2,None)
        kp2, des2 = orb.compute(im2, kp2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        if len(matches) == 0:
            return 0
        else:
            good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
            return len(good)+0.0 / len(matches)
    except:
        return 0

def make_matches(locs_i,locs_j,args):
    max_sim = 0
    path_i = os.path.join(args.dataset_dir,"c0"+str(locs_i[0][0]),"vdo.avi")
    path_j = os.path.join(args.dataset_dir,"c0"+str(locs_j[0][0]),"vdo.avi")
    vid_i = cv2.VideoCapture(path_i)
    vid_j = cv2.VideoCapture(path_j)
    for i in range(locs_i.shape[0]):
        for j in range(locs_j.shape[0]):
            vid_i.set(cv2.CAP_PROP_POS_FRAMES,locs_i[i][2])
            vid_j.set(cv2.CAP_PROP_POS_FRAMES,locs_j[j][2])
            _,im_i = vid_i.read()
            _,im_j = vid_j.read()
            im_i = im_i[locs_i[i][4]:locs_i[i][4]+locs_i[i][6],locs_i[i][3]:locs_i[i][3]+locs_i[i][5]]
            im_j = im_j[locs_j[j][4]:locs_j[j][4]+locs_j[j][6],locs_j[j][3]:locs_j[j][3]+locs_j[j][5]]
            w = max(im_i.shape[0],im_j.shape[0])
            h = max(im_i.shape[1],im_j.shape[1])
            im_i = cv2.resize(im_i, (w, h), interpolation=cv2.INTER_CUBIC)
            im_j = cv2.resize(im_j, (w, h), interpolation=cv2.INTER_CUBIC)
            sim = orb_sim(im_i,im_j)
            if sim>max_sim:
                max_sim = sim
    return max_sim



def prepare_loc_base(res):
    locs = []
    for line in res:
        line = line.strip()
        tmp = line.split(" ")
        locs.append(list(map(int, tmp)))
    return np.vstack(locs)

def shuffle_locs(locs,args):
    sf_locs = []
    cams = np.unique(locs[:,0])
    for cam in cams:
        cam_locs = locs[locs[:,0]==cam]
        tracks = np.unique(cam_locs[:,1])
        for track in tracks:
            track_loc = cam_locs[cam_locs[:,1]==track]
            stride = track_loc.shape[0]//args.shuffle_len
            for i in range(args.shuffle_len):
                sf_locs.append(track_loc[i*stride])
    return np.vstack(sf_locs)
    

def find_matches(sf_locs,args):
    reid_dict = {}
    cams = np.unique(sf_locs[:,0])
    for cam in cams:
        reid_dict[cam] = {}
    for i in range(len(cams)):
        cami = cams[i]
        cami_locs = sf_locs[sf_locs[:,0]==cami]
        tracksi = np.unique(cami_locs[:,1])
        for tracki in tracksi:
            reid_dict[cami][tracki] = tracki

    for i in range(len(cams)-1):
        for j in range(i+1,len(cams)):
            cami = cams[i]
            camj = cams[j]
            cami_locs = sf_locs[sf_locs[:,0]==cami]
            camj_locs = sf_locs[sf_locs[:,0]==camj]
            tracksi = np.unique(cami_locs[:,1])
            tracksj = np.unique(camj_locs[:,1])
            for tracki in tracksi:
                max_sim = 0
                max_j = -1
                for trackj in tracksj:
                    locs_i = cami_locs[cami_locs[:,1]==tracki]
                    locs_j = camj_locs[camj_locs[:,1]==trackj]
                    sim = make_matches(locs_i,locs_j,args)
                    if sim > max_sim:
                        max_sim = sim
                        max_j = trackj
                if max_sim>args.sim_min:
                    reid_dict[camj][max_j] = tracki
    return reid_dict

def re_id(reid_dict,or_tracks,f):
    for line in or_tracks:
        line = line.strip().split(" ")
        line[1] = str(reid_dict[int(line[0])][int(line[1])])
        f.write(" ".join(line)+"\n")


def main(args):
    with open(args.file,"r") as f:
        or_tracks = f.readlines()
    locs = prepare_loc_base(or_tracks)
    sf_locs = shuffle_locs(locs,args)
    reid_dict = find_matches(sf_locs,args)
    g = open(args.output,"w")
    re_id(reid_dict,or_tracks,g)
    




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="fast_reid")
    parser.add_argument(
        '--dataset_dir',default="/mnt/hdda/kevinq/aic_20_trac3/test/S06", help='Path to dataset directory')
    parser.add_argument(
        '--shuffle_len',default=4,type=int, help='shuffle length')
    parser.add_argument(
        '--sim_min', default = 70.0,type = float,help='min sim of reid')
    parser.add_argument(
        '--file', default="/mnt/hdda/kevinq/aic_20_trac3/test/output.txt",help='Path to input track')
    parser.add_argument(
        '--output', default="/mnt/hdda/kevinq/aic_20_trac3/track3.txt",help='output path')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
    
        


                    












    