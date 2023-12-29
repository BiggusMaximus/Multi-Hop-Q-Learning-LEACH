from get_all_routes import get_best_nodes
from get_all_routes import get_best_net
from get_all_routes import get_all_best_routes
from get_all_routes import get_cost
from get_all_routes import count_routes
from collections import Counter

import numpy as np
import copy
import random
                                                        
def update_Q(T,Q,current_state, next_state, alpha):
    current_t = T[current_state][next_state]
    current_q = Q[current_state][next_state]
    new_q = current_q + alpha * (current_t + min(Q[next_state].values()) - current_q)
    print(f"Q-update | current state : {current_state} | Q[current_state] : {Q[current_state]} | next_state : {next_state} | Q[next_state] : {Q[next_state]}")
    Q[current_state][next_state] = new_q   
    return Q


def get_min_state(dic,valid_moves):
    """input dic is like {3: -0.5, 10: -0.1}
    valid_moves is like [1,3,5]"""
    new_dict = dict((k, dic[k]) for k in valid_moves)
    return min(new_dict, key=new_dict.get)


def get_route(Q,start,end):
    """ input is  Q-table is like:{1: {2: 0.5, 3: 3.8},
                                   2: {1: 5.9, 5: 10}} """   
    single_route = [start]
    while single_route[-1] not in end:
        next_step = min(Q[single_route[-1]],key=Q[single_route[-1]].get)
        single_route.append(next_step)
        if len(single_route) > 2 and single_route[-1] in single_route[:-1]:
            break
    return single_route

def get_key_of_min_value(dic):
        min_val = min(dic.values())
        return [k for k, v in dic.items() if v == min_val]

def get_key_of_random_value(dic):
        return [k for k, v in dic.items()]

def Q_routing(T,Q,alpha,epsilon,n_episodes,start,end):
    nodes_number = [0,0]
    for e in range(n_episodes):
        print(e)
        if e in range(0,n_episodes,1000):
            print("loop:",e)
        current_state = start
        #current_route = [start]
        goal = False
        while not goal:
            valid_moves = list(Q[current_state].keys())
            print(f"Valid : {valid_moves}")
            if len(valid_moves) <= 1:
                next_state = valid_moves[0]
            else:
                if random.random() < epsilon:
                    best_action = np.min(get_key_of_min_value(Q[current_state]))
                    print(f"best_action : {best_action}")
                    valid_moves.pop(valid_moves.index(best_action))
                    print(f"Valid pop  : {valid_moves}")
                    next_state = random.choice(valid_moves)
                    print(f"next_state  : {next_state}")
                
                else:
                    next_state = random.choice(get_key_of_min_value(Q[current_state]))
                    print(f"best_action : {next_state}")
            Q = update_Q(T,Q,current_state, next_state, alpha)
        
            if next_state in end:
                goal = True
            current_state = next_state
            #current_route.append(next_state)
        #print "current:",current_route   
        #print get_route(Q,start,end)
        # check stop standard
        if e in range(0, 1000, 50):
            for i in Q.keys():
                for  j in Q[i].keys():
                    Q[i][j]  = round(Q[i][j],6)
            print("Q-Learning Nodes ")
            nodes = get_best_nodes(Q,start,end)
            nodes_number.append(len(nodes))
            print("================= ")
            if len(set(nodes_number[-3:])) == 1:
                break
    return Q
    
