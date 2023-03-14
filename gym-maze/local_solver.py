import sys
import numpy as np
import math
import random
import json
import requests
import time
import gym
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from riddle_solvers import *

vis = np.zeros((10, 10))
pro = np.zeros((10, 10, 4))
prevStates = []
visRiddles = np.zeros((4))
lastAction = 0
path = []
idxpath = 0
count = 0
total = 0
endFlag = 0
riddlesCount = 0


def valid(prei, prej, i, j, action):
    global vis
    global pro
    return (
        i >= 0
        and i <= 9
        and j >= 0
        and j <= 9
        and vis[i][j] == 0
        and pro[prei][prej][action] == 0
    )


def select_action(state):
    # print(state[0])
    global riddlesCount
    global path
    global lastAction
    global prevStates
    global vis
    global pro
    global count
    global total
    global endFlag
    riddlesCount = len(state[-1])
    total += 1
    actions = ["S", "E","N", "W"]
    directions = [[0, 1], [1, 0],[0, -1], [-1, 0]]
    # print(lastAction)
    # print(state[0])
    # This is a random agent

    if state[0][0] == 9 and state[0][1] == 9:
        path.clear()
        path.append(state[0].copy())
        endFlag = 1

    if endFlag and count == riddlesCount:

        for i in range(4):
            newi = state[0][0] + directions[i][0]
            newj = state[0][1] + directions[i][1]
            if newi == 9 and newj == 9:

                return actions[i], i
        for i in range(4):
            newi = state[0][0] + directions[i][0]
            newj = state[0][1] + directions[i][1]
            if newi == path[-1][0] and newj == path[-1][1]:

                path.pop()
                return actions[i], i

    elif endFlag and (
        len(path) == 0
        or not (path[-1][0] == state[0][0] and path[-1][1] == state[0][1])
    ):
        path.append(state[0].copy())

    vis[state[0][0]][state[0][1]] = 1
    # print("last: ",prevStates)

    if (
        len(prevStates) > 0
        and prevStates[-1][0] == state[0][0]
        and prevStates[-1][1] == state[0][1]
    ):
        pro[state[0][0]][state[0][1]][lastAction] = 1

    # print("before: ",prevStates)
    if len(prevStates) == 0 or not (
        prevStates[-1][0] == state[0][0] and prevStates[-1][1] == state[0][1]
    ):
        prevStates.append(state[0].copy())
    # print("after: ",prevStates)
    # This function should get actions from your trained agent when inferencing.

    for i in range(4):
        newi = state[0][0] + directions[i][0]
        newj = state[0][1] + directions[i][1]

        if valid(state[0][0], state[0][1], newi, newj, i):
            lastAction = i
            return actions[i], i

    prevStates.pop()
    # print(state[0],count)
    for i in range(4):
        newi = state[0][0] + directions[i][0]
        newj = state[0][1] + directions[i][1]
        if newi == prevStates[-1][0] and newj == prevStates[-1][1]:
            prevStates.pop()
            return actions[i], i


def local_inference(riddle_solvers):
    global count
    global total
    global riddlesCount
    global visRiddles
    global path
    obv = manager.reset(agent_id)

    good=1

    for t in range(MAX_T):

        # Select an action
        state_0 = obv
        action, action_index = select_action(state_0)  # Random action
        obv, reward, terminated, truncated, info = manager.step(agent_id, action)

        if not info["riddle_type"] == None:
            solution = riddle_solvers[info["riddle_type"]](info["riddle_question"])
            obv, reward, terminated, truncated, info = manager.solve_riddle(
                info["riddle_type"], agent_id, solution
            )
            print(obv)

        for i in range(len(state_0[-1])):
            if state_0[-1][i][0] == 0 and state_0[-1][i][1] == 0 and visRiddles[i] == 0:
                count += 1
                visRiddles[i] = 1
        
        if count == riddlesCount:
            if len(path) == 0:
                #print(0.8 * 4000 / total)
                #print(4000 / ( total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))))
                #print("---------------")
                if 0.8 * 4000 / total > 4000 / ( total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))):
                    #print("total1: ", total) 
                    manager.set_done(agent_id)
                    break  # Stop Agent
            elif good:
                #print(0.8 * 4000 / total)
                #print(4000 / ( total + len(path)))
                if 0.8 * 4000 / total > 4000 / ( total + len(path)):
                    #print("total2: ", total)    
                    manager.set_done(agent_id)
                    break  # Stop Agent
                else:
                    good=0

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if np.array_equal(obv[0], (9, 9)) and count == riddlesCount:
            #print(4000 / total)
            #print("total3: ", total)
            manager.set_done(agent_id)
            break  # Stop Agent

        if RENDER_MAZE:
            manager.render(agent_id)
        states[t] = [
            obv[0].tolist(),
            action_index,
            str(manager.get_rescue_items_status(agent_id)),
        ]


if __name__ == "__main__":

    sample_maze = np.load("Sample7.npy")
    agent_id = "9"  # add your agent id here

    manager = MazeManager()
    manager.init_maze(agent_id, maze_cells=sample_maze)
    env = manager.maze_map[agent_id]

    riddle_solvers = {
        "cipher": cipher_solver,
        "captcha": captcha_solver,
        "pcap": pcap_solver,
        "server": server_solver,
    }
    maze = {}
    states = {}

    maze["maze"] = env.maze_view.maze.maze_cells.tolist()
    maze["rescue_items"] = list(manager.rescue_items_dict.keys())

    MAX_T = 5000
    RENDER_MAZE = True

    local_inference(riddle_solvers)

    with open("./states.json", "w") as file:
        json.dump(states, file)

    with open("./maze.json", "w") as file:
        json.dump(maze, file)
