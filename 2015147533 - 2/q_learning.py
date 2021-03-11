"""
The original code is from https://github.com/dennybritz/reinforcement-learning/tree/master/TD
"""
import ipdb
import sys
import numpy as np
import itertools
import pickle
from collections import defaultdict
from game import Game


# In our case, we have 3 action (stay, go-left, go-right)
def get_action_num():
    return 3


## this function return policy function to choose the action based on Q value.
def make_policy(Q, epsilon, nA):
    """
    This is the epsilon-greedy policy, which select random actions for some chance (epsilon).
    (Check dennybritz's repository for detail)

    You may change the policy function for the given task.
    """
    def policy_fn(observation):        
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def reachability(basket_loc, target_locs):
    itembox=[[0,0],[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[4,5],[5,6],[6,7],[7,8],[7,8]]
    filtered = []
    for tl in target_locs:
        if(itembox[tl[1]][0]>basket_loc):
            if np.abs(itembox[tl[1]][0] - basket_loc) <= 9 - tl[2]:
                filtered.append(tl)
        elif(itembox[tl[1]][1]<basket_loc):
            if np.abs(itembox[tl[1]][1] - basket_loc) <= 9 - tl[2]:
                filtered.append(tl)
        elif(itembox[tl[1]][0]<=basket_loc and itembox[tl[1]][1]>=basket_loc):
            filtered.append(tl)
        #ipdb.set_trace()
    return filtered

## this function return state from given game information.
def get_state(counter, score, game_info):
    basket_location, item_location = game_info
    """
    FILL HERE!
    you can (and must) change the return value.
    """
    coin_locs = [item for item in item_location if item[0] == 1]
    coin_locs = reachability(basket_location, coin_locs)
    #coin_dists = [np.linalg.norm(np.array([item[1],item[2]]) - np.array([basket_location,9])) for item in coin_locs]
    clock_locs = [item for item in item_location if item[0] == 2]
    clock_locs = reachability(basket_location, clock_locs)
    #clock_dists = [np.linalg.norm(np.array([item[1],item[2]]) - np.array([basket_location,9])) for item in clock_locs]

    itembox=[[0,0],[0,1],[0,2],[1,2],[2,3],[3,4],[4,5],[4,5],[5,6],[6,7],[7,8],[7,8]]

    if len(coin_locs) == 0:
        coin_loc_array = np.array([]).reshape([0,3])
    else:
        coin_loc_array = np.array(coin_locs)

    if len(clock_locs) == 0:
        clock_loc_array = np.array([]).reshape([0,3])
    else:
        clock_loc_array = np.array(clock_locs)


    
    coin_state = (coin_loc_array  - np.array([0, basket_location, 9])).tolist()
    clock_state = (clock_loc_array  - np.array([0, basket_location, 9])).tolist()

    if (len(coin_state) > 0 and len(clock_state)>0):
        coinx=coin_loc_array[0]
        clockx=clock_loc_array[0]
        if(itembox[coinx[1]][0]<=basket_location and basket_location<=itembox[coinx[1]][1]):
          coinmovedir=0
        elif(itembox[coinx[1]][1]<basket_location):
           coinmovedir=1
        elif(itembox[coinx[1]][0]>basket_location):
            coinmovedir=-1

        if(itembox[clockx[1]][0]<=basket_location and basket_location<=itembox[clockx[1]][1]):
            clockmovedir=0
        elif(itembox[clockx[1]][1]<basket_location):
            clockmovedir=1
        elif(itembox[clockx[1]][0]>basket_location):
            clockmovedir=-1

        if coinmovedir > 0 and clockmovedir >0:
            final_state = '0'
        elif coinmovedir > 0 and clockmovedir==0:
            final_state = '1'
        elif coinmovedir > 0 and clockmovedir <0:
            final_state = '2'
        elif coinmovedir == 0 and clockmovedir >0:
            final_state = '3'
        elif coinmovedir == 0 and clockmovedir == 0:
            final_state = '4'
        elif coinmovedir == 0 and clockmovedir <0:
            final_state = '5'
        elif coinmovedir < 0 and clockmovedir >0:
            final_state = '6'
        elif coinmovedir < 0 and clockmovedir ==0:
            final_state = '7'
        elif coinmovedir < 0 and clockmovedir <0:
            final_state = '8'

        
    elif(len(coin_state)> 0 and len(clock_state)<=0):
        coinx=coin_loc_array[0]
        if(itembox[coinx[1]][0]<=basket_location and basket_location<=itembox[coinx[1]][1]):
          coinmovedir=0
        elif(itembox[coinx[1]][1]<basket_location):
           coinmovedir=1
        elif(itembox[coinx[1]][0]>basket_location):
            coinmovedir=-1
        
        if coinmovedir > 0:
            final_state = '9'
        elif coinmovedir == 0:
            final_state = '10'
        elif coinmovedir < 0:
            final_state = '11'

    elif(len(coin_state)<=0 and len(clock_state)>0):
        clockx=clock_loc_array[0]
        if(itembox[clockx[1]][0]<=basket_location and basket_location<=itembox[clockx[1]][1]):
            clockmovedir=0
        elif(itembox[clockx[1]][1]<basket_location):
            clockmovedir=1
        elif(itembox[clockx[1]][0]>basket_location):
            clockmovedir=-1
        if clockmovedir > 0:
            final_state = '12'
        elif clockmovedir == 0:
            final_state = '13'
        elif clockmovedir < 0:
            final_state = '14'
    else:
        final_state ='-1'


    #print(state)
    return final_state



## this function return reward from given previous and current score and counter.
def get_reward(prev_score, current_score, prev_counter, current_counter):
    """
    FILL HERE!
    you can (and must) change the return value.
    (current_score /current_counter) - (prev_score/ prev_counter)
    """
    score =0
    if((current_counter - prev_counter) > 0):
        
        if((current_score-prev_score)==0):
            score =10000
        else:
            score = 5000
    else:
            if((current_score-prev_score)>0):
                score = 100
            else:
                score =-100
            
    return score 


def save_q(Q, num_episode, params, filename="model_q.pkl"):
    data = {"num_episode": num_episode, "params": params, "q_table": dict(Q)}
    with open(filename, "wb") as w:
        w.write(pickle.dumps(data))

        
def load_q(filename="model_q.pkl"):
    with open(filename, "rb") as f:
        data = pickle.loads(f.read())
        return defaultdict(lambda: np.zeros(3), data["q_table"]), data["num_episode"], data["params"]


def q_learning(game, num_episodes, params):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy.
    You can edit those parameters, please speficy your changes in the report.
    
    Args:
        game: Coin drop game environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        Q: the optimal action-value function, a dictionary mapping state -> action values.
    """
    
    epsilon, alpha, discount_factor = params
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(get_action_num()))  
    
    # The policy we're following
    policy = make_policy(Q, epsilon, get_action_num())
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        _, counter, score, game_info = game.reset()
        state = get_state(counter, score, game_info)
        action = 0
        
        # One step in the environment
        for t in itertools.count():
            # Take a step
            action_probs = policy(get_state(counter, score, game_info))
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            done, next_counter, next_score, game_info = game.step(action)
            
            next_state = get_state(counter, score, game_info) 
            reward = get_reward(score, next_score, counter, next_counter)
            counter = next_counter
            score = next_score
            
            """
            this code performs TD Update. (Update Q value)
            You may change this part for the given task.
            """
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
        
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("Episode {}/{} (Score: {})\n".format(i_episode + 1, num_episodes, score), end="")
            sys.stdout.flush()

    return Q

def train(num_episodes, params):
    g = Game(False)
    Q = q_learning(g, num_episodes, params)
    return Q


## This function will be called in the game.py
def get_action(Q, counter, score, game_info, params):
    epsilon = params[0]
    policy = make_policy(Q, epsilon, 3)
    action_probs = policy(get_state(counter, score, game_info))
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_episode", help="# of the episode (size of training data)",
                    type=int, required=True)
    parser.add_argument("-e", "--epsilon", help="the probability of random movement, 0~1",
                    type=float, default=0.1)
    parser.add_argument("-lr", "--learning_rate", help="learning rate of training",
                    type=float, default=0.1)
    
    args = parser.parse_args()
    
    if args.num_episode is None:
        parser.print_help()
        exit(1)

    # you can pass your parameter as list or dictionary.
    # fix corresponding parts if you want to change the parameters
    
    num_episodes = args.num_episode
    epsilon = args.epsilon
    learning_rate = args.learning_rate
    
    Q = train(num_episodes, [epsilon, learning_rate, 0.5])
    save_q(Q, num_episodes, [epsilon, learning_rate, 0.5])
    
    Q, n, params = load_q()

if __name__ == "__main__":
    main()
