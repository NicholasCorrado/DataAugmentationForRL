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