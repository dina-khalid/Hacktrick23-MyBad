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
from queue import Queue


actions = ["S", "E", "N", "W"]
directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
vis = np.zeros((10, 10))
pro = np.zeros((10, 10, 4))
distances = np.zeros((10, 10, 4))
prevStates = []
visRiddles = np.zeros((4))
parent = np.zeros((10, 10, 2))
path = []
lastAction = 0
count = 0
total = 0
endFlag = 0
riddlesCount = 0
bfsArray = np.zeros((10, 10))


def valid(prei, prej, i, j, action):
    global vis
    global pro
    return (
        i >= 0
        and i <= 9
        and j >= 0
        and j <= 9
        and vis[i][j] < 3
        and pro[prei][prej][action] != 1
    )


def validBFS(prei, prej, i, j, action):
    global vis
    global pro
    global bfsArray
    return (
        i >= 0
        and i <= 9
        and j >= 0
        and j <= 9
        and vis[i][j] >= 1
        and pro[prei][prej][action] == 0
        and bfsArray[i][j] > bfsArray[prei][prej] + 1
    )


def sumH(state):
    h = 0
    for i in range(0, len(state[-1])):
        h += state[i]
    return h


def BFS(x, y):
    global actions
    global directions
    global bfsArray
    for i in range(10):
        for j in range(10):
            bfsArray[i][j] = 1e9
    bfsArray[x][y] = 0

    q = Queue(maxsize=200)
    q.put([x, y])
    while not q.qsize() == 0:
        node = q.get()
        print(node)
        print(q.qsize())
        for i in range(4):
            newi = node[0] + directions[i][0]
            newj = node[1] + directions[i][1]
            print(f"{newi} - {newj}")
            if validBFS(node[0], node[1], newi, newj, i):
                bfsArray[newi][newj] = bfsArray[node[0]][node[1]] + 1
                q.put([newi, newj])


def getBFSPath(start, end):
    global actions
    global directions
    global path
    for i in range(10):
        for j in range(10):
            bfsArray[i][j] = 1e9
    bfsArray[start[0]][start[1]] = 0

    q = Queue(maxsize=100)
    q.put((start[0], start[1]))
    print("nodeeeeeeeeeeeee")
    while q.qsize() != 0:
        node = q.get()
        print(node)
        print(f"size {q.qsize()}")
        for i in range(4):
            newi = node[0] + directions[i][0]
            newj = node[1] + directions[i][1]
            print(f"{newi} - {newj} - {validBFS(node[0], node[1], newi, newj, i)}")
            if validBFS(node[0], node[1], newi, newj, i):
                bfsArray[newi][newj] = bfsArray[node[0]][node[1]] + 1
                parent[newi][newj] = node
                q.put([newi, newj])

    if end[0] == 9 and end[1] == 9:
        print(vis.T)
        print(bfsArray.T)

    path.clear()
    currentNodeX = int(end[0])
    currentNodeY = int(end[1])
    print("mm")
    print(start)
    while currentNodeX != start[0] or currentNodeY != start[1]:
        if currentNodeX or currentNodeY:
            print(currentNodeX)
            print(currentNodeY)
        path.append((currentNodeX, currentNodeY))
        tmp = parent[currentNodeX][currentNodeY]
        currentNodeX = int(tmp[0])
        currentNodeY = int(tmp[1])
    path = path[::-1]


def getDistances(x, y, rescues):
    distance = 0
    minimumD=1e9
    for i in range(len(rescues)):
        if rescues[i] == -1:
            continue
        distance += distances[x][y][i]
        minimumD=min(distance,distances[x][y][i])
    return distance+minimumD


def select_action(state):
    # print(state[0])
    global riddlesCount
    global lastAction
    global prevStates
    global vis
    global pro
    global count
    global total
    global endFlag
    global actions
    global directions
    global path
    riddlesCount = len(state[-1])
    total += 1

    # print("last: ",prevStates)

    if count == riddlesCount:
        if not endFlag:
            endFlag = 1
            print("endflag")
            getBFSPath(state[0], [9, 9])

        for i in range(4):
            newi = state[0][0] + directions[i][0]
            newj = state[0][1] + directions[i][1]
            if newi == path[-1][0] and newj == path[-1][1]:
                path.pop()
                return actions[i], i

    if not vis[state[0][0]][state[0][1]]:
        for i in range(len(state[-2])):
            distances[state[0][0]][state[0][1]][i] = state[-2][i]

    if len(prevStates) > 0:
        if prevStates[-1][0] == state[0][0] and prevStates[-1][1] == state[0][1]:
            pro[state[0][0]][state[0][1]][lastAction] = 1
            pro[state[0][0] + directions[lastAction][0]][
                state[0][1] + directions[lastAction][1]
            ][(lastAction + 2) % 4] = 1
        else:
            pro[prevStates[-1][0]][prevStates[-1][1]][lastAction] = 0
            pro[state[0][0]][state[0][1]][(lastAction + 2) % 4] = 0

    # print("before: ",prevStates)
    if len(prevStates) == 0 or not (
        prevStates[-1][0] == state[0][0] and prevStates[-1][1] == state[0][1]
    ):
        prevStates.append(state[0].copy())
    # print("after: ",prevStates)

    # calculate sum of h for the state

    vis[state[0][0]][state[0][1]] += 1

    # print(lastAction)
    # print(state[0])
    # This is a random agent
    minimumAns = total + getDistances(state[0][0], state[0][1], state[-2]) - 10
    minimumNode = state[0]
    print(state[0])
    if len(path) == 0:
        BFS(state[0][0], state[0][1])
        for i in range(10):
            for j in range(10):
                if not vis[i][j]:
                    continue
                # print(i, end=" ")
                # print(j, end=" ")
                # print(bfsArray[i][j], end=" ")
                # print(getDistances(i, j, state[-2]))
                tmpAns = (
                    total
                    + bfsArray[i][j]
                    + getDistances(i, j, state[-2])
                    + 5 * vis[i][j]
                )
                if tmpAns < minimumAns:
                    minimumAns = tmpAns
                    minimumNode = [i, j]

        getBFSPath(state[0], minimumNode)

    # print("--")
    # print(state[0])
    # print(path)
    if len(path) > 0:
        for i in range(4):
            newi = state[0][0] + directions[i][0]
            newj = state[0][1] + directions[i][1]
            if newi == path[-1][0] and newj == path[-1][1]:
                path.pop()
                return actions[i], i

    # This function should get actions from your trained agent when inferencing.

    minimumAns = 1e9
    minimumNode = state[0]
    actionIndex = -1
    action = "N"
    # print("*******")
    for i in range(4):
        newi = state[0][0] + directions[i][0]
        newj = state[0][1] + directions[i][1]
        print(newi, end=" ")
        print(newj, end=" ")
        print(valid(state[0][0], state[0][1], newi, newj, i))
        if valid(state[0][0], state[0][1], newi, newj, i):
            tmpAns = (
                total + 1 + getDistances(newi, newj, state[-2]) + 5 * vis[newi][newj]
            )
            # print(getDistances(newi, newj, state[-2]), end=" ")
            # print(newi, end=" ")
            # print(newj, end=" ")
            # print(vis[newi][newj])
            if tmpAns < minimumAns:
                minimumAns = tmpAns
                minimumNode = [newi, newj]
                action = actions[i]
                actionIndex = i
    # print("action: ", actionIndex)
    if actionIndex != -1:
        lastAction = actionIndex
        return action, actionIndex

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
    global pro
    for i in range(10):
        for j in range(10):
            pro[i][j] = -1
    obv = manager.reset(agent_id)

    good = 1

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
                # print(0.8 * 4000 / total)
                # print(4000 / ( total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))))
                # print("---------------")
                if 0.8 * 4000 / total > 4000 / (
                    total + 3.333333333 * ((9 - state_0[0][0]) + (9 - state_0[0][1]))
                ):
                    # print("total1: ", total)
                    manager.set_done(agent_id)
                    break  # Stop Agent
            elif good:
                # print(0.8 * 4000 / total)
                # print(4000 / ( total + len(path)))
                if 0.8 * 4000 / total > 4000 / (total + len(path)):
                    # print("total2: ", total)
                    manager.set_done(agent_id)
                    break  # Stop Agent
                else:
                    good = 0

        # THIS IS A SAMPLE TERMINATING CONDITION WHEN THE AGENT REACHES THE EXIT
        # IMPLEMENT YOUR OWN TERMINATING CONDITION
        if np.array_equal(obv[0], (9, 9)) and count == riddlesCount:
            # print(4000 / total)
            # print("total3: ", total)
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

    sample_maze = np.load("Sample6.npy")
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
