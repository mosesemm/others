
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN, A2C
from sklearn.metrics import confusion_matrix

from fire_modeling_env import FireModelingEnv
from dataset_generator import data_set, WIDTH, HEIGHT
from commons import MODEL_PATH, argmax
import numpy as np
from visualize import save_animation


env = FireModelingEnv(data_set)

env = DummyVecEnv([lambda : FireModelingEnv(data_set)])

NUM_ACTIONS_TESTED = 10

def get_actual(position):
    record = [curr for curr in data_set if (curr[0], curr[1]) == position][0]
    return record[4]

def fig_to_img(fig):
    return np.array(fig.canvas.renderer._renderer)

def evaluate_model(model):
    obs = env.reset()
    frames = []
    frames.append(fig_to_img(env.render()))

    actual = []
    predicted = []
    for i in range(NUM_ACTIONS_TESTED):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        frames.append(fig_to_img(env.render()))

        predicted.append(True)
        actual.append(get_actual((obs[0][0], obs[0][1])))

    mat = confusion_matrix(actual, predicted)

    return frames, mat





def evaluate_ppo(env):
    model = PPO2(MlpPolicy, env, verbose=1)
    model.load(MODEL_PATH)

    return evaluate_model(model)

'''
frames, mat = evaluate_ppo(env)
print("*******Confusion matrix (PPO): ")
print(mat)
save_animation(frames, "ppo_test")
'''


def evaluate_model_free(q_values):
    obs = env.reset()
    frames = []
    frames.append(fig_to_img(env.render()))
    actual = []
    predicted = []

    def get_mod_free_action(obs):
        lookup_action_values = q_values["{}_{}".format(obs[0][0], obs[0][1])]
        #print(lookup_action_values)
        best_action = argmax(lookup_action_values)
        return best_action

    for i in range(NUM_ACTIONS_TESTED):
        action = get_mod_free_action(obs)
        obs, rewards, done, info = env.step([action])
        frames.append(fig_to_img(env.render()))

        predicted.append(True)
        actual.append(get_actual((obs[0][0], obs[0][1])))


    mat = confusion_matrix(actual, predicted)

    return frames, mat
