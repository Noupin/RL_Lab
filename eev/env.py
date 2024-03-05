import numpy as np
import random
import tensorflow as tf
import keras

# Global parameters
initial_energy = 10.0
energy_cost_of_action = 0.1
decay_to_heat_conversion = 0.5
full_energy_conversion_rate = 0.8
energy_cost_of_reproduction = 2.0  # Energy cost for reproduction
partial_energy_conversion_rate = 0.5
partial_energy_loss_rate = 0.6
energy_cost_of_eating = 0.2
mutation_rate = 0.005  # 0.5% chance of mutation


class Cell:
  def __init__(self, action_weights=None, position=(0, 0)):
    if action_weights is None:
      # Randomly initialize action weights for a new cell
      self.action_weights = np.random.rand(7)  # 7 actions
      self.action_weights /= self.action_weights.sum() * 7  # Normalize to sum to 7
    else:
      # Inherit action weights from parent
      self.action_weights = action_weights

    self.energy = initial_energy
    self.network = self.create_network()
    self.size = random.uniform(0.5, 1.0)
    self.is_dead = False
    self.max_size = 10
    self.position = position

  def create_network(self):
    model = keras.Sequential([
      keras.layers.Dense(units=10, activation='relu',
                         input_shape=(10,)),  # 10 inputs
      keras.layers.Dense(units=7, activation='softmax')  # 4 actions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

  def decide_action(self):
    touch_data = self.check_surroundings()
    input_data = np.array([self.energy] + touch_data +
                          list(self.action_weights))
    input_data = np.reshape(input_data, (1, -1))
    # ... rest of the decide_action method ...

  def check_surroundings(self):
    # Logic to check for adjacent cells in each direction
    # Returns a list of booleans [up, down, left, right]
    return [False, False, False, False]  # Placeholder

  def act(self, environment):
    action = self.decide_action()
    if action.startswith('move_'):
      self.move(action, environment)
    elif action == 'eat':
      self.eat(environment)
    elif action == 'reproduce':
      self.reproduce(environment)
    elif action == 'dock':
      self.dock(environment)
    # ... rest of the act method ...

  def move(self, direction, environment):
    # Implement movement logic based on direction
    # ...
    pass

  def eat(self, environment):
    # Implement eating logic
    # ...
    pass

  def reproduce(self, environment):
    if self.energy > energy_cost_of_reproduction:
      child_weights = self.action_weights.copy()
      if random.random() < mutation_rate:
        child_weights += np.random.uniform(-0.05, 0.05, 4)
        child_weights /= child_weights.sum() * 4
      child_position = self.position  # Modify for different spawn location
      new_cell = Cell(action_weights=child_weights, position=child_position)
      environment.cells.append(new_cell)
      self.energy -= energy_cost_of_reproduction

  def dock(self, environment):
    # Implement docking logic
    # ...
    pass

  def die(self):
    self.is_dead = True


class EevEnvironment:
  def __init__(self, size, initial_cell_count):
    self.size = size
    self.cells = [Cell(position=(random.randint(
      0, size - 1), random.randint(0, size - 1))) for _ in range(initial_cell_count)]
    self.heat = 0

  def step(self):
    for cell in self.cells:
      cell.act(self)
      self.apply_rules(cell)

    self.remove_dead_cells()

    if self.has_enough_energy():
      self.cells.append(Cell(position=(random.randint(
        0, self.size - 1), random.randint(0, self.size - 1))))

  def apply_rules(self, cell):
    cell.size -= cell.decay_rate
    if cell.size <= 0:
      cell.alive = False

  def remove_dead_cells(self):
    self.cells = [cell for cell in self.cells if cell.alive]

  def has_enough_energy(self):
    # Logic to determine if the environment can support a new cell
    return random.choice([True, False])  # Example condition


# Example usage
env = EevEnvironment(size=100, initial_cell_count=10)
for _ in range(100):
  env.step()
