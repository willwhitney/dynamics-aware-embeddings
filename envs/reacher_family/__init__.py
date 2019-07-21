from gym.envs.registration import registry, register, make, spec

from reacher_family.reacher_push import ReacherPushEnv
from reacher_family.reacher_vertical import ReacherVerticalEnv
from reacher_family.reacher_turn import ReacherTurnEnv


register(
    id='ReacherVertical-v2',
    entry_point='reacher_family:ReacherVerticalEnv',
    max_episode_steps=100,
)

register(
    id='ReacherPush-v2',
    entry_point='reacher_family:ReacherPushEnv',
    max_episode_steps=100,
)

register(
    id='ReacherTurn-v2',
    entry_point='reacher_family:ReacherTurnEnv',
    max_episode_steps=100,
)
