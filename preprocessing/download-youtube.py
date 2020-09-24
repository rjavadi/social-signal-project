from fabric import Connection
from fabric import Task
import pandas as pd
import string
import subprocess
import os

df = pd.read_csv('../data/yt_dataset.csv')

video_tuples = [tuple(row) for row in df.values]

cn = Connection('rjavadi@127.0.0.1', connect_kwargs={
        "password": "incantatem",
    })
dir = '/home/rjavadi/PycharmProjects/contempt-affective-computing/data'
#https://stackoverflow.com/questions/31514949/ffmpeg-over-https-fails

# with cn.cd(dir):
#     for (i, (link, label, duration)) in enumerate(video_tuples):
#         if i>7:
#             res = cn.run('pwd')
#             end_of_link = link.find('?')
#             skip_index = link.find('=')
#             skip_time = link[skip_index + 1:]
#             link = link[:end_of_link]
#             print('link: ', link, 'skip time:', skip_time)
#             cmd = "/home/rjavadi/PycharmProjects/contempt-affective-computing/ffmpeg-4.3-amd64-static/ffmpeg -ss {skip}  -i \"$(/home/rjavadi/tmp/youtube-dl  -g \"{link}\" | head -1)\" -t {duration} -c copy -f avi {index}.avi".format(skip = skip_time, link = link, duration = duration, index ='other_'+str(i+1))
#             print(cmd)
#             result = cn.run(cmd)

for (i, (link, label, duration)) in enumerate(video_tuples):
    end_of_link = link.find('?')
    skip_index = link.find('=')
    skip_time = link[skip_index + 1:]
    link = link[:end_of_link]
    print('link: ', link, 'skip time:', skip_time)
    cmd = "/home/rjavadi/PycharmProjects/contempt-affective-computing/ffmpeg-4.3-amd64-static/ffmpeg -ss {skip}  -i \"$(/home/rjavadi/tmp/youtube-dl  -g \"{link}\" | head -1)\" -t {duration} -c copy -f avi {index}.avi".format(skip = skip_time, link = link, duration = duration, index =label+'_'+str(i+1))
    # print(cmd)
    subprocess.call(cmd, shell=True)
        

        # /home/rjavadi/PycharmProjects/contempt-affective-computing/ffmpeg-4.3-amd64-static/ffmpeg -ss 838  -i "$(/home/rjavadi/tmp/youtube-dl  -g " https://youtu.be/opSDn1t2IeY" | head -1)" -t 10 -c copy -f avi to_be_cut.avi


# Download youtube movie piece:
# 
# First acquire the actual URL using youtube-dl:
# 
# youtube-dl -g "https://www.youtube.com/watch?v=V_f2QkBdbRI"
# 
# 
# Copy the output of the command and paste it as part of the -i parameter of the next command:
# 
# ffmpeg -ss 00:00:15.00 -i "OUTPUT-OF-FIRST URL" -t 00:00:10.00 -c copy out.mp4
# 
# 
# The -ss parameter in this position states to discard all input up until 15 seconds into the video. The -t option states to capture for 10 seconds. The rest of the command tells it to store as an mp4.
# 
# ffmpeg -ss 00:10:15.00 -i "$(./youtube-dl  -g "https://www.youtube.com/watch?v=HxxHCgp9hHc" | head -1)" -t 00:00:05.00 -c copy out.mp4
# 
# ffmpeg -ss 545  -i "$(./youtube-dl  -g "https://www.youtube.com/watch?v=HxxHCgp9hHc" | head -1)" -t 5 -c copy out.mp4
