import argparse
import json
import logging
import os
import os.path as osp
import sys

import pandas as pd
import cv2
from ..pipeline import task
from ..system import VideoJob, VideoSystem
from . import log

logger = log.get_logger(__name__)

def update_track(track,args,vid_size_dict,video_id):
    w = track[3]-track[1]
    h = track[4]-track[2]
    new_x1 = track[1] - args.expand*w
    new_x2 = track[3] + args.expand*w
    new_y1 =track[2] - args.expand*h
    new_y2 = track[4] + args.expand*h
    new_x1 = max(1,int(new_x1))
    new_x2 = min(int(new_x2),vid_size_dict[video_id][0])
    new_y1 = max(1,int(new_y1))
    new_y2 = min(int(new_y2),vid_size_dict[video_id][1])
    return [track[0],new_x1,new_y1,new_x2,new_y2]
    

def get_jobs(args):
    with open(args.video_list_file) as f:
        lines = f.readlines()
    jobs = []
    vid_dict = {}
    vid_size_dict = {}
    for i in range(len(lines)):
        line = lines[i]
        video_name = line.strip()
        camera_id = int(line.strip().split("/")[-2].split("c")[-1])
        video_cap = cv2.VideoCapture(os.path.join(args.dataset_dir,line.strip()))
        n_frames = int(video_cap.get(7))
        video_id = i
        # print(video_name, video_id, camera_id, n_frames)
        vid_dict[video_id] = camera_id
        vid_size_dict[str(camera_id)] = [video_cap.get(cv2.CAP_PROP_FRAME_WIDTH),video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
        jobs.append(VideoJob(video_name, video_id, camera_id, n_frames))
    return jobs,vid_dict,vid_size_dict

def write_output(f,output,args,vid_list,vid_size_dict):
    #simple remove negatvie predctions
    negative_list = []
    for e in output:
        video_id = str(vid_list[e[0]])
        track_id = str(e[1])
        tracks = e[-1]
        if len(tracks)<args.min_track_length:
            continue
        else:
            for track in tracks:
                if int(track[1])<=0 or int(track[2])<=0 or int(track[3]-track[1])<=1 or int(track[4]-track[2])<=1:
                    tag = "c"+video_id+"t"+track_id
                    negative_list.append(tag)
                    break   

    for e in output:
        video_id = str(vid_list[e[0]])
        track_id = str(e[1])
        tracks = e[-1]
        if len(tracks)<args.min_track_length:
            continue
        if "c"+video_id+"t"+track_id in negative_list:
            continue
        else:
            for track in tracks:
                # track = update_track(track,args,vid_size_dict,video_id)
                # f.write(video_id+" "+track_id+" "+str(track[0])+" "+str(max(int(track[1]),0))+" "+str(max(int(track[2]),0))+" "+str(max(int(track[3]-track[1]),1))+" "+str(max(int(track[4]-track[2]),1))+" 0 0\n")
                f.write(video_id+" "+track_id+" "+str(track[0])+" "+str(int(track[1]))+" "+str(int(track[2]))+" "+str(int(track[3]-track[1]))+" "+str(int(track[4]-track[2]))+" 0 0\n")





def main(args):
    logger.info('Running with args: %s', args)
    # print(args)
    profiling = 'profiling' in os.environ
    if profiling:
        logger.info('Running in profiling mode')
    os.makedirs(osp.dirname(args.system_output), exist_ok=True)
    f = open(args.system_output,"w")
    jobs,vid_dict,vid_size_dict = get_jobs(args)
    system = VideoSystem(args.dataset_dir, args.cache_dir, stride=args.stride)
    logger.info('Running %d jobs', len(jobs))
    if len(jobs) > 0:
        show_progress = not args.silent
        system.init(
            jobs[0], timeout=args.init_timeout,
            show_progress=show_progress, print_result=profiling)
    for job in jobs:
        system.process(
            [job], 1, timeout=args.frame_timeout,
            show_progress=show_progress, print_result=profiling * 30)
        output = system.get_output()
        write_output(f,output,args,vid_dict,vid_size_dict)
        # output.to_csv(args.system_output, sep=' ', header=None, index=False)
    system.finish()
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0]))
    parser.add_argument(
        'dataset_dir', help='Path to dataset directory')
    parser.add_argument(
        'video_list_file', help='Path to video list file')
    parser.add_argument(
        'system_output', help='Path to output (output.json)')
    parser.add_argument(
        '--cache_dir', default=None,
        help='Path to save intermediate results, if provided')
    parser.add_argument(
        '--stride', default=1, type=int,
        help='Stride of processing frames (default: 1)')
    parser.add_argument(
        '--silent', action='store_true', help='Silent frame level progressbar')
    parser.add_argument(
        '--init_timeout', default=600, type=int,
        help='Timeout of system initialization (default: 60)')
    parser.add_argument(
        '--frame_timeout', default=100, type=int,
        help='Timeout of processing 1 frame (default: 10)')
    parser.add_argument(
        '--min_track_length', default=20, type=int,
        help='min length of tracking output (default: 10)')
    parser.add_argument(
        '--expand', default=0, type=float,
        help='expand rate of tracking box (default: 0.05)')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
