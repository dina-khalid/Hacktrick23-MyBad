from gym_maze.envs.maze_manager import MazeManager
import gym_maze
import gym
import time
import sys
import numpy as np
import math
import random
import json
import requests

from riddle_solvers import *

# the api calls must be modified by you according to the server IP communicated with you
# students track --> 16.170.85.45
# working professionals track --> 13.49.133.141
server_ip = '16.170.85.45'


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
    actions = ["S", "E", "N", "W"]
    directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
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


def move(agent_id, action):
    response = requests.post(
        f'http://{server_ip}:5000/move', json={"agentId": agent_id, "action": action})
    return response


def solve(agent_id,  riddle_type, solution):
    response = requests.post(f'http://{server_ip}:5000/solve', json={
                             "agentId": agent_id, "riddleType": riddle_type, "solution": solution})
    print(response.json())
    return response


def get_obv_from_response(response):
    directions = response.json()['directions']
    distances = response.json()['distances']
    position = response.json()['position']
    obv = [position, distances, directions]
    return obv


def submission_inference(riddle_solvers):
    global count
    global total
    global riddlesCount
    global visRiddles
    global path

    response = requests.post(
        f'http://{server_ip}:5000/init', json={"agentId": agent_id})
    obv = get_obv_from_response(response)
    good = 1

    while(True):
        # Select an action
        state_0 = obv
        action, action_index = select_action(state_0)  # Random action
        response = move(agent_id, action)
        if not response.status_code == 200:
            print(response)
            break

        obv = get_obv_from_response(response)
        print(response.json())

        if not response.json()['riddleType'] == None:
            solution = riddle_solvers[response.json()['riddleType']](
                response.json()['riddleQuestion'])
            response = solve(agent_id, response.json()['riddleType'], solution)

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        for i in range(len(state_0[-1])):
            if state_0[-1][i][0] == 0 and state_0[-1][i][1] == 0 and visRiddles[i] == 0:
                count += 1
                visRiddles[i] = 1

        if count == riddlesCount:
            if len(path) == 0:
                #print(0.8 * 4000 / total)
                #print(4000 / ( total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))))
                # print("---------------")
                if 0.8 * 4000 / total > 4000 / (total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))):
                    #print("total1: ", total)
                    response = requests.post(
                        f'http://{server_ip}:5000/leave', json={"agentId": agent_id})
                    break  # Stop Agent
            elif good:
                #print(0.8 * 4000 / total)
                #print(4000 / ( total + len(path)))
                if 0.8 * 4000 / total > 4000 / (total + len(path)):
                    #print("total2: ", total)
                    response = requests.post(
                        f'http://{server_ip}:5000/leave', json={"agentId": agent_id})
                    break  # Stop Agent
                else:
                    good = 0

        if np.array_equal(response.json()['position'], (9, 9)) and count == riddlesCount:
            response = requests.post(
                f'http://{server_ip}:5000/leave', json={"agentId": agent_id})
            break


if __name__ == "__main__":

    agent_id = "h6vGyCqcVZ"
    riddle_solvers = {'cipher': cipher_solver, 'captcha': captcha_solver,
                      'pcap': pcap_solver, 'server': server_solver}
    submission_inference(riddle_solvers)


def createTokens():
    private_key = jwk.JWK.generate(kty="RSA", size=2048)
    private_key_pem = private_key.export_to_pem(
        private_key=True, password=None)
    public_key_pem = private_key.export_to_pem()
    with open('private_key.pem', 'wb') as f:
        f.write(private_key_pem)

    with open('public_key.pem', 'wb') as f:
        f.write(public_key_pem)
