import os
import glob
from src.dafs.panda import push, slide, pickandplace, flip

DAFS = {
    'PandaPush-v3': {
        'RelabelGoal': push.RelabelGoal,
    },
    'PandaSlide-v3': {
        'RelabelGoal': slide.RelabelGoal,
    },
    'PandaPickAndPlace-v3': {
        'RelabelGoal': pickandplace.RelabelGoal,
    },
    'PandaFlip-v3': {
        'RelabelGoal': flip.RelabelGoal,
    }
}

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id
