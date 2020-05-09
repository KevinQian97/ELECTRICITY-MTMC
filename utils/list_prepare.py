import os
Base_path = "/mnt/hdda/kevinq/aic_20_trac3/validation/S05"
f = open("/mnt/hdda/kevinq/aic_20_trac3/validation/file-list.txt","w")
cams = os.listdir(Base_path)
for cam in cams:
    f.write(os.path.join(cam,"vdo.avi")+"\n")
    