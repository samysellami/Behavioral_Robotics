import gym
import numpy as np
env = gym.make('CartPole-v0')

pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons
# the number of inputs and outputs depends on the problem
# we assume that observations consist of vectors of continuous value
# and that actions can be vectors of continuous values or discrete actions
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0

sigma = 0.02 # mutation step size
lmbda = 10 #population size (should be an even number)
theta = np.zeros((nhiddens*ninputs + noutputs*nhiddens + nhiddens + noutputs, lmbda))
thta_len = nhiddens*ninputs + noutputs*nhiddens + nhiddens + noutputs

for i in range(lmbda):
    W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
    W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
    b1 = np.zeros(shape=(nhiddens, 1)) # bias first layer
    b2 = np.zeros(shape=(noutputs, 1)) # bias second layer
    thta = np.vstack((np.resize(W1,(ninputs * nhiddens,1)), np.resize(W2,(noutputs * nhiddens,1)), b1, b2))  # parameters of the population
    theta[:,i] =  thta[:,0]


# function to extract the training parameters from the vector of parameters of the population 
def extract_params(theta,j):
    idx = 0
    idx1 = ninputs * nhiddens
    W1 = np.resize( theta[idx:idx1, j], (nhiddens,ninputs))

    idx = ninputs * nhiddens
    idx1 = ninputs * nhiddens + noutputs * nhiddens
    W2 = np.resize( theta[idx:idx1 , j], (noutputs , nhiddens))

    idx = ninputs * nhiddens + noutputs * nhiddens
    idx1 = ninputs * nhiddens + noutputs * nhiddens + nhiddens
    b1 = np.resize( theta[idx:idx1, j], (nhiddens,1))

    idx = ninputs * nhiddens + noutputs * nhiddens + nhiddens
    idx1 = ninputs * nhiddens + noutputs * nhiddens + nhiddens + noutputs 
    b2 = np.resize( theta[idx:idx1, j], (noutputs,1))
    return W1, W2, b1, b2



for i in range(200):
    # initialize the fitness vector
    s = np.zeros((lmbda))

    for j in range(lmbda):
	# extract the training parameters 
        W1, W2, b1, b2 = extract_params(theta, j)
        
        observation = env.reset()

        for _ in range(500):
        
                # convert the observation array into a matrix with 1 column and ninputs rows
            observation.resize(ninputs,1)
                # compute the netinput of the first layer of neurons
            Z1 = np.dot(W1, observation) + b1
                # compute the activation of the first layer of neurons with the tanh function
            A1 = np.tanh(Z1)
                # compute the netinput of the second layer of neurons
            Z2 = np.dot(W2, A1) + b2
                # compute the activation of the second layer of neurons with the tanh function
            A2 = np.tanh(Z2)
                # if actions are discrete we select the action corresponding to the most activated unit
            if (isinstance(env.action_space, gym.spaces.box.Box)):
                action = A2
            else:
                action = np.argmax(A2)   
            
            #env.render()
            observation, reward, done, info = env.step(action)
            # update the fitness value for the evaluation episode i	
            s[j] = s[j] +  reward

        env.close()
    print(s)
    # ranking individuals by fitness
    u = np.argsort(-s)
	
    # updating the parameters vectors according to fitness ranking
    for l in range(int(lmbda/2)):
        eps = np.random.randn(thta_len,1) * sigma
        theta[:, u[ int(lmbda/2) + l] ] = theta[:, u[l]]   +  eps[:,0]

    # extract the best parameters for the current evaluation episode
    W1, W2, b1, b2 = extract_params(theta, u[0])        
    observation = env.reset()
    
    # render the environment for the best parameters of the current evalutaion episode
    for _ in range(500):
        
            # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(ninputs,1)
            # compute the netinput of the first layer of neurons
        Z1 = np.dot(W1, observation) + b1
            # compute the activation of the first layer of neurons with the tanh function
        A1 = np.tanh(Z1)
            # compute the netinput of the second layer of neurons
        Z2 = np.dot(W2, A1) + b2
            # compute the activation of the second layer of neurons with the tanh function
        A2 = np.tanh(Z2)
            # if actions are discrete we select the action corresponding to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)   
        
        env.render()
        observation, reward, done, info = env.step(action)

