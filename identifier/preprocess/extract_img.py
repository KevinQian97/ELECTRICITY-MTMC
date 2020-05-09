import os
import cv2
from multiprocessing import Pool
import argparse
import shutil

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tracklet_path', type=str, default="./exp/tracklets.txt",
                        help='path to the tracklets')
    parser.add_argument('--data_path', type=str, default="./datasets/aic_20_trac3",
                        help='path to the aicity 2020 track 3 folders')
    parser.add_argument('--output_path', type=str, default="./exp/imgs/aic_test",
                        help='path to the output dictionaries')
    parser.add_argument("--njobs",type = int,default=3,help="number of pools to extract imgs")
    return parser

def sort_tracklets(gts):
    sorted_gts = {}
    car_list = []
    for line in gts:
        line = line.strip().split(" ")[:-2]
        frame = int(line[2])
        left = int(line[3])
        top = int(line[4])
        right = left +int(line[5])
        bot = top+int(line[6])
        car_id = int(line[1])
        query = False
        if car_id not in car_list:
            car_list.append(car_id)
            query = True
        if frame not in list(sorted_gts.keys()):
            sorted_gts[frame] = []
        sorted_gts[frame].append([left,top,right,bot,car_id,query])
    print(len(car_list))
    return sorted_gts

    

def extract_im_api(args):
    base_path = args[0]
    data_path = args[1]
    split = args[2]
    scene = args[3]
    cam = args[4]
    gts = args[5]
    extrac_im(base_path,data_path,split,scene,cam,gts)

def extrac_im(base_path,data_path,split,scene,cam,gts):
    print("start cam:"+cam)
    scene_dir = os.path.join(base_path,split,scene)
    cam_dir = os.path.join(scene_dir,cam)
    cap = cv2.VideoCapture(os.path.join(cam_dir,"vdo.avi"))
    sorted_gts = sort_tracklets(gts)
    fr_id=0
    state,im = cap.read()
    frames = list(sorted_gts.keys())
    while(state):
        if fr_id not in frames or im is None:
            state,im = cap.read()
            fr_id+=1
        else:
            tracks = sorted_gts[fr_id]
            for track in tracks:
                left,top,right,bot,car_id,query=track
                clip = im[top:bot,left:right]
                im_name = str(car_id).zfill(5)+"_"+cam+"_"+str(fr_id).zfill(4)+".jpg"
                if query:
                    if not os.path.exists(os.path.join(data_path,"image_query")):
                        os.makedirs(os.path.join(data_path,"image_query"))
                    cv2.imwrite(os.path.join(data_path,"image_query",im_name),clip)      
                else:
                    if not os.path.exists(os.path.join(data_path,"image_test")):
                        os.makedirs(os.path.join(data_path,"image_test"))
                    cv2.imwrite(os.path.join(data_path,"image_test",im_name),clip) 
            state,im = cap.read()
            fr_id+=1


def main(args):
    gts_path = args.tracklet_path
    base_path = args.data_path
    data_path = args.output_path
    if os.path.exists(os.path.join(data_path)):
        shutil.rmtree(os.path.join(data_path)) 
    os.makedirs(os.path.join(data_path))
    splits = ["test"]

    with open(gts_path,"r") as f:
        or_tracks = f.readlines()
    args_list = []
    for split in splits:
        split_dir = os.path.join(base_path,split)
        scenes = os.listdir(split_dir)
        for scene in scenes:
            scene_dir = os.path.join(split_dir,scene)
            cams = os.listdir(scene_dir)
            for cam in cams:
                gts = []
                camid = int(cam.split("c")[1])
                for track in or_tracks:
                    if int(track.split(" ")[0]) == camid:
                        gts.append(track)
                args_list.append([base_path,data_path,split,scene,cam,gts])
    n_jobs = args.njobs
    pool = Pool(n_jobs)
    pool.map(extract_im_api, args_list)
    pool.close()



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)