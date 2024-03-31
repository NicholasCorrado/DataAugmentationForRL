import src.dafs.panda
import src.dafs.panda.push
import src.dafs.panda.slide
import src.dafs.panda.pickandplace
import src.dafs.panda.flip

import src.dafs.nav2d
import src.dafs.pointmaze
import src.dafs.antmaze

DAFS = {
    'PandaPush-v3': {
        'RelabelGoal': panda.push.RelabelGoal,
    },
    'PandaSlide-v3': {
        'RelabelGoal': panda.slide.RelabelGoal,
    },
    'PandaPickAndPlace-v3': {
        'RelabelGoal': panda.pickandplace.RelabelGoal,
    },
    'PandaFlip-v3': {
        'RelabelGoal': panda.flip.RelabelGoal,
    },

    'Nav2d-v0': {
        'TranslateAgent': nav2d.TranslateAgent
    },

    'PointMaze_UMaze-v3': {
        'TranslateRotate': pointmaze.TranslateRotate,
        'RelabelGoal': pointmaze.RelabelGoal,
        'TranslateRotateRelabelGoal': pointmaze.TranslateRotateRelabelGoal,
    },


}
