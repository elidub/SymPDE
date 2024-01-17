import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
import io
from tqdm import tqdm
import re

def get_data(filter = '', logdir = '../logs'):
    name_dirs = glob.glob(os.path.join(logdir) + '/*/*')
    data = {'_'.join(os.path.normpath(name_dir).split('/')[-2:]) : {'dir' : name_dir} for name_dir in name_dirs}

    # fitler data.keys() on regex
    pattern = re.compile(filter)
    for key in list(data.keys()):
        if not pattern.search(key):
            data.pop(key)

    return data

def get_metrics(version_dir, metrics):

    metrics = {metric : [] for metric in metrics}

    filename_events = [filename for filename in os.listdir(version_dir) if 'events.out.tfevents' in filename]
    # assert len(filename_events) == 1, filename_events
    # filename_event = filename_events[0]
    # filename_event = filename_events[-1]
    for filename_event in filename_events:
        raw_dataset = tf.data.TFRecordDataset(os.path.join(version_dir, filename_event))

        for raw_record in raw_dataset:
            event = tf.compat.v1.Event.FromString(raw_record.numpy())
            for v in event.summary.value:
                for metric in metrics:
                    if v.tag == metric:
                        if v.HasField('simple_value'):
                            metrics[metric].append(v.simple_value)
                        elif v.HasField('image'):
                            pass
    
    return metrics

def get_results(data, metrics = ['val_loss', 'test_loss']):
    for version, version_dict in tqdm(data.items()):
        metrics = get_metrics(version_dict['dir'], metrics = metrics)
        version_dict.update(metrics)
    return data