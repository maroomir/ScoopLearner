import numpy as np
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy

best_mean_reward, n_steps = -np.inf, 0

# Create log dir
log_dir = "logs/"

os.makedirs(log_dir, exist_ok=True)

def getBestRewardCallback():
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    def bestRewardCallback(_locals, _globals, model=None):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward
        # Print stats every 1 calls
        divider = 1
        
        if (n_steps + 1) % divider == 0 and (n_steps + 1) / divider > 1:
        # if True:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    # Example for saving best model
                    print("Saving new best model",best_mean_reward)
                    if model is not None:
                        model.save(log_dir + str(n_steps) +'_best_model.pkl')
                    else:
                        _locals['self'].save(log_dir + str(n_steps) +'_best_model.pkl')
                    
                if model is not None:
                    model.save(log_dir + '/model.pkl')
                else:
                    _locals['self'].save(log_dir + '/model.pkl')
       

        n_steps += 1

        return True

    return bestRewardCallback

def logDir():
    return log_dir
