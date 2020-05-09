import pandas as pd


def get_output(events):
    output = [(e.video_id, e.track_id, e.track)
              for e in events]
    # output = pd.DataFrame(output, columns=[
    #     'video_id', 'frame_id', 'obj_type',"track_id","track"])
    return output
