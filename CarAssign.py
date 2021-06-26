
import gym
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from collections import namedtuple
from time import sleep


Right_CMD = [0, 1]
Left_CMD = [1, 0]

# Defining the reward configuration
Game_Evolve = 10

# Defining the game positions
Game_actions = [
    [1, 0, 0],  # Movement 0
    [0, 1, 0],  # Movement 1
    [0, 0, 1]   # Movement 2
]

# importing the MountainCar-v0 Game Environment from OpenAI Gym
env = gym.make('MountainCar-v0')

# Defining the structure to store the game and reward data
GameData = namedtuple('GameData', 'reward data')

# function to compute the positions
def compute_reward(pos):
    """
    Compute Reward for Current Position.
    :param position:
    :return:
    """
    # Update Best Position
    if pos >= -0.1000000:
        return 6
    if pos >= -0.1100000:
        return 5
    if pos >= -0.1300000:
        return 4
    if pos >= -0.1500000:
        return 3
    if pos >= -0.1700000:
        return 2
    if pos >= -0.2000000:
        return 1

    return -1


def play_random_games(Games=1000):
    """
    function for playing random games to get some observations

    """

    # array to store all games movements
    Movements = []
    # loop to play 1000 games
    for Episode in range(Games):

        # setting game reward as 0
        Episode_Rwd = 0

        # array to store current game data/ creating memory to store current game data
        Curr_Game_mem = []

        # resetting game environment
        env.reset()

        # getting first random movements
        action = env.action_space.sample()
        
        # if the action occurs
        while True:

            # Playing the game
            #observation is the position of the agent
            # reward will be the reward in each steps in each episode
            observation, reward, done, info = env.step(action)  # observation=position, velocity

            # updating the reward value using compute_reward function
            reward = compute_reward(observation[[0]])
            
            #getting the next movement to compare with previous movement using random action
            action = env.action_space.sample()

            # Storing observations and actions taken during random game in a horizontal stack
            Curr_Game_mem.append(
                np.hstack((observation, Game_actions[action]))
            )

            if done:
                break
            # calculating rewards in each episode
            Episode_Rwd += reward

        # Computing the reward
        if (Episode_Rwd > -199.0):
            print(f'Reward={Episode_Rwd}')

            # saving all data in Movements array
            Movements.append(
                GameData(Episode_Rwd, Curr_Game_mem)
            )

    # Sorting the Movements array to get best N games
    Movements.sort(key=lambda item: item.reward, reverse=True)

    # Filtering the best N games
    Movements = Movements[Game_Evolve] if len(Movements) > Game_Evolve else Movements

    # Retrieve only the best game movements
    Mvm_Only = []
    for Single_Game in Movements:
        Mvm_Only.extend([item for item in Single_Game.data])

    # Creating DataFrame to store the best N games
    dataframe = pd.DataFrame(
        Mvm_Only,
        columns=['pos', 'velocity', 'action_0', 'action_1', 'action_2']
    )
    return dataframe


def generate_ml(dataframe):
    """
    To Train and Generate Neural network Model
    """
    # Define Neural Network architecture
    model = Sequential()
    #input layer
    model.add(Dense(3, input_dim=2, activation='relu'))
    #hidden layer 1
    model.add(Dense(32,  activation='relu'))
    # hidden layer 2
    model.add(Dense(32,  activation='relu'))
    #output layer
    model.add(Dense(3, activation='sigmoid'))

    # Compiling Neural Network model
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Fitting the Model with Data
    model.fit(
        dataframe[['pos', 'velocity']],
        dataframe[['action_0', 'action_1', 'action_2']],
        epochs=80
    )

    return model


def play_game(ml_model, games=10):
    """
    Play te Game
    """
    # loop to play 10 games
    for i_epi in range(games):

        # Defining the reward variable
        epi_Reward = 0

        # resetting the environment to play the real game
        observation = env.reset()
        
        while True:
            # rendring the environment
            render = env.render()
           
            #Connection of the game to the network
            # Predicting the Next Movement using the neural network
            current_action_pred = ml_model.predict(observation.reshape(1, 2))[0]

            # Defining the Movement
            current_action = np.argmax(current_action_pred)

            # Making the Movement
            observation, reward, done, info = env.step(current_action)

            # Updating the Reward Value
            epi_Reward += compute_reward(observation[[0]])
            
            # printing the episode withe reward 
            if done:
                print(f"Episode finished after {i_epi+1} steps", end='')
                break

        print(f" Score = {epi_Reward}")

if __name__ == '__main__':
    #calling the function for random playby passing the number of games to play
    print("---Playing Random Games---")
    df = play_random_games(Games=1000)
    
    # calling the function to train the network model
    print("---Training NN Model---")
    ml_model = generate_ml(df)
    
    #calling the function to play real game
    print("---Playing Games with NN---")
    play_game(ml_model=ml_model, games=10)