import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm  # For progress bar

# Environment class
class TruckRoutingEnvironment:
    def __init__(self, points, demands, truck_capacity, origin_node=0):
        self.points = points
        self.demands = demands
        self.truck_capacity = truck_capacity
        self.origin_node = origin_node
        self.current_location = origin_node
        self.current_capacity = truck_capacity
        self.pending_demands = demands.copy()
        self.visited = [False] * len(points)
        self.visited[origin_node] = True

    def reset(self):
        self.current_location = self.origin_node
        self.current_capacity = self.truck_capacity
        self.pending_demands = self.demands.copy()
        self.visited = [False] * len(self.points)
        self.visited[self.origin_node] = True
        return self._get_state()

    def step(self, action):
        if action == len(self.points):  # Return to origin
            distance = self._calculate_distance(self.current_location, self.origin_node)
            self.current_location = self.origin_node
            self.current_capacity = self.truck_capacity
            reward = -distance  # Penalize for traveling back to origin
        else:
            distance = self._calculate_distance(self.current_location, action)
            delivered = min(self.pending_demands[action], self.current_capacity)
            self.pending_demands[action] -= delivered
            self.current_capacity -= delivered
            self.current_location = action
            self.visited[action] = True
            reward = delivered - distance  # Reward for delivery minus travel cost

        done = all(demand == 0 for demand in self.pending_demands)
        return self._get_state(), reward, done

    def _get_state(self):
        return np.array([
            *self.points[self.current_location],
            self.current_capacity,
            *self.pending_demands,
            *self.visited
        ])

    def _calculate_distance(self, p1, p2):
        return np.sqrt((self.points[p1][0] - self.points[p2][0])**2 + 
                       (self.points[p1][1] - self.points[p2][1])**2)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0]))
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function with progress bar
def train_dqn_agent(env, episodes, batch_size):
    state_size = len(env.reset())
    action_size = len(env.points) + 1  # All points + return to origin
    agent = DQNAgent(state_size, action_size)
    
    rewards = []
    for e in tqdm(range(episodes), desc="Training Progress"):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        rewards.append(total_reward)
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, rewards

# Visualization functions
def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def visualize_route(env, agent):
    state = env.reset()
    done = False
    route = [env.current_location]
    
    while not done:
        action = agent.act(state)
        state, _, done = env.step(action)
        route.append(env.current_location)
    
    points = np.array(env.points)
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c='blue')
    for i, (x, y) in enumerate(points):
        plt.annotate(str(i), (x, y))
    
    route_points = points[route]
    plt.plot(route_points[:, 0], route_points[:, 1], c='red', linewidth=2, zorder=2)
    plt.title('Truck Route')
    plt.show()

def animate_route(env, agent):
    fig, ax = plt.subplots(figsize=(10, 10))
    points = np.array(env.points)
    scatter = ax.scatter(points[:, 0], points[:, 1], c='blue')
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y))
    
    line, = ax.plot([], [], c='red', linewidth=2, zorder=2)
    truck, = ax.plot([], [], 'go', markersize=10, zorder=3)

    def init():
        line.set_data([], [])
        truck.set_data([], [])
        return line, truck

    def update(frame):
        if frame == 0:
            env.reset()
        
        state = env._get_state()
        action = agent.act(state)
        _, _, done = env.step(action)

        route = np.array([env.points[loc] for loc in range(len(env.points)) if env.visited[loc]])
        line.set_data(route[:, 0], route[:, 1])
        truck.set_data(env.points[env.current_location])

        if done:
            anim.event_source.stop()
        
        return line, truck

    anim = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=200)
    plt.title('Animated Truck Route')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Use the same parameters as before
    n = 100
    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    demands = [random.randint(1, 10) for _ in range(n)]
    demands[0] = 0  # Origin has no demand

    env = TruckRoutingEnvironment(points, demands, truck_capacity=5, origin_node=0)
    trained_agent, rewards = train_dqn_agent(env, episodes=3, batch_size=32)

    # Plot learning curve
    plot_learning_curve(rewards)

    # Visualize a single route
    visualize_route(env, trained_agent)

    # Animate the route
    animate_route(env, trained_agent)

    # Save the trained model
    trained_agent.model.save('truck_routing_dqn_model.h5')
