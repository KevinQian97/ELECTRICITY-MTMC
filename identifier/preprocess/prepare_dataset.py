import os
import cv2
base_path = "./datasets/aic_20_trac3"
data_path = "./datasets/Aic"
splits = ["train","validation"]
from multiprocessing import Pool
import shutil


def sort_tracklets(gts):
    sorted_gts = {}
    car_list = []
    for line in gts:
        line = line.strip().split(",")[:-4]
        frame = int(line[0])
        left = int(line[2])
        top = int(line[3])
        right = left +int(line[4])
        bot = top+int(line[5])
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
    extrac_im(base_path,data_path,split,scene,cam)

def extrac_im(base_path,data_path,split,scene,cam):
    print("start cam:"+cam)
    print(split)
    scene_dir = os.path.join(base_path,split,scene)
    cam_dir = os.path.join(scene_dir,cam)
    cap = cv2.VideoCapture(os.path.join(cam_dir,"vdo.avi"))
    with open(os.path.join(cam_dir,"gt","gt.txt"),"r") as f:
        gts = f.readlines()
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
                    elif split =="train":
                        if not os.path.exists(os.path.join(data_path,"image_train")):
                            os.makedirs(os.path.join(data_path,"image_train"))
                        cv2.imwrite(os.path.join(data_path,"image_train",im_name),clip)
                    else:
                        if not os.path.exists(os.path.join(data_path,"image_test")):
                            os.makedirs(os.path.join(data_path,"image_test"))
                        cv2.imwrite(os.path.join(data_path,"image_test",im_name),clip) 
                state,im = cap.read()
                fr_id+=1


def main():
    args_list = []
    for split in splits:
        split_dir = os.path.join(base_path,split)
        scenes = os.listdir(split_dir)
        for scene in scenes:
            scene_dir = os.path.join(split_dir,scene)
            cams = os.listdir(scene_dir)
            for cam in cams:
                args_list.append([base_path,data_path,split,scene,cam])
    n_jobs = 10
    pool = Pool(n_jobs)
    pool.map(extract_im_api, args_list)
    pool.close()

    #we resplit the present training and validation set to have more tranining samples
    train_path = os.path.join(data_path,"image_train")
    test_path = os.path.join(data_path,"image_test")
    test_list = ["c006","c007","c008","c009"]
    imgs = os.listdir(test_path)
    for img in imgs:
        cam_id = img.split("_")[1]
        if cam_id not in test_list:
            shutil.move(os.path.join(test_path,img),os.path.join(train_path,img))



if __name__ == "__main__":
    main()
                    




                



