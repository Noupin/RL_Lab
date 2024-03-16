import numpy as np
import random
import tensorflow as tf
import keras
import os

np.set_printoptions(suppress=True)


def clear_screen():
  # For Windows
  if os.name == 'nt':
    for _ in range(10):
      print("\n")
  # For macOS and Linux
  else:
    _ = os.system('clear')


# Global parameters
initial_energy = 1.0
energy_cost_of_action = 0.1
decay_to_heat_conversion = 0.5
full_energy_conversion_rate = 0.8
energy_cost_of_reproduction = 2.0  # Energy cost for reproduction
partial_energy_conversion_rate = 0.5
partial_energy_loss_rate = 0.6
energy_cost_of_eating = 0.2
mutation_rate = 0.005  # 0.5% chance of mutation
raycast_distance = 1  # Distance for raycasting
size_ratio_threshold = 0.75
base_energy_transfer = 5.0
environment_size = 10
starting_cell_count = 2


class Cell:
  def __init__(self, action_weights=None, position=(0, 0)):
    # For new cells or cells that just so happen to mutate
    if action_weights is None or np.random.rand() < mutation_rate:
      # Randomly initialize action weights for a new cell
      # 4 actions(there are 7 but move is only counting as 1)
      self.action_weights = np.random.rand(4)
      # Normalize to sum to 4
      self.action_weights *= (4 / self.action_weights.sum())
    else:
      # Inherit action weights from parent with slight variance
      variance = np.random.normal(0, 0.05, 4)  # Small random variance
      self.action_weights = action_weights + variance
      # Normalize the weights
      self.action_weights *= (4 / self.action_weights.sum())

    self.energy = initial_energy
    self.network = self.create_network()
    self.size = random.uniform(0.5, 1.0)
    self.decay_rate = 0.1
    self.is_dead = False
    self.max_size = 10
    self.position = position
    self.direction = None  # Initialize direction
    self.is_docked = False

  def create_network(self):
    model = keras.Sequential([
        keras.layers.Dense(units=6, activation='relu',
                           input_shape=(6,)),  # 10 inputs
        keras.layers.Dense(units=7, activation='softmax')  # 7 actions
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

  def decide_action(self, environment: "EevEnvironment") -> str:
    # Get raycast information
    is_within_ray = self.cast_ray(environment)
    input_data = np.array(
      [self.energy] + [is_within_ray] + list(self.action_weights))
    input_data = np.reshape(input_data, (1, -1))

    action_probabilities = self.network.predict(input_data, verbose=0)[0]
    action = np.random.choice(['move_up', 'move_down', 'move_left',
                              'move_right', 'eat', 'reproduce', 'dock'], p=action_probabilities)
    return action

  def cast_ray(self, environment: "EevEnvironment"):
    # Cast a ray in the direction the cell is facing
    return environment.check_ray(self.position, self.direction, raycast_distance)

  def get_front_position(self):
    x, y = self.position
    if self.direction == "up":
      y -= 1
    elif self.direction == "down":
      y += 1
    elif self.direction == "left":
      x -= 1
    elif self.direction == "right":
      x += 1
    return x, y

  def act(self, environment: "EevEnvironment"):
    action: str = self.decide_action(environment)
    # print(action)
    match action:
      case action if action.startswith('move_'):
        self.move(action, environment)
      case 'eat':
        self.eat(environment)
      case 'reproduce':
        self.reproduce(environment)
      case 'dock':
        self.dock(environment)
        # TODO: Add fix dock if two multi organisms are docked

  def move(self, direction, environment: "EevEnvironment"):
    movement_distance = int(self.action_weights[0])

    if self.is_docked:
      # If part of a multicellular organism, handle collective movement
      self.handle_multicell_movement(direction, movement_distance, environment)
    else:
      # Individual cell movement
      self.execute_individual_movement(
        direction, movement_distance, environment)

  def execute_individual_movement(self, direction, movement_distance, environment: "EevEnvironment"):
    # Calculate new position based on direction and movement distance
    new_x, new_y = self.calculate_new_position(
      direction, movement_distance, environment)
    # Update position and environment grid
    self.update_position((new_x, new_y), environment)

  def handle_multicell_movement(self, direction, movement_distance, environment: "EevEnvironment"):
    # Calculate new position for the organism
    new_x, new_y = self.calculate_new_position(
      direction, movement_distance, environment)

    # Find the organism to which this cell belongs
    organism_root = environment.find(self)
    # Apply this movement to each cell in the organism
    for cell in environment.get_organism_cells(organism_root):
      cell.update_position((new_x, new_y), environment)

  def calculate_new_position(self, direction, movement_distance, environment: "EevEnvironment"):
    x, y = self.position
    match direction:
      case 'move_up':
        y -= movement_distance
      case 'move_down':
        y += movement_distance
      case 'move_left':
        x -= movement_distance
      case 'move_right':
        x += movement_distance
    # Ensure the new position is within environment bounds
    x = max(0, min(x, environment.size - 1))
    y = max(0, min(y, environment.size - 1))
    return x, y

  def update_position(self, new_position, environment: "EevEnvironment"):
    old_pos = self.position
    self.position = new_position
    environment.update_grid(self, old_pos)

  def eat(self, environment: "EevEnvironment"):
    eating_efficiency = self.action_weights[1]  # Eating weight
    front_position = self.get_front_position()
    target_cell = environment.get_cell_at_position(front_position)

    if target_cell:
      energy_gained = self.calculate_energy_gained(
        target_cell, eating_efficiency)

      if self.is_docked:
        # Split the energy: some for this cell, the rest distributed among the organism
        self_energy = energy_gained * 0.3  # 30% of the energy goes to this cell
        shared_energy = energy_gained * 0.7  # 70% is shared with the organism

        organism_cells = environment.get_organism_cells(environment.find(self))
        shared_energy_per_cell = shared_energy / len(organism_cells)

        for cell in organism_cells:
          if cell != self:
            cell.energy += shared_energy_per_cell
        self.energy += self_energy
      else:
        # Individual cell energy update
        self.energy += energy_gained

      if target_cell.energy <= 0:
        target_cell.die()
        environment.remove_cell(target_cell)

  def calculate_energy_gained(self, target_cell: "Cell", eating_efficiency):
    size_ratio = self.size / target_cell.size

    if size_ratio >= size_ratio_threshold:
      # Full consumption logic
      energy_transfer = min(target_cell.energy, base_energy_transfer *
                            eating_efficiency * full_energy_conversion_rate)
    else:
      # Partial consumption logic
      energy_transfer = min(target_cell.energy, base_energy_transfer *
                            eating_efficiency * partial_energy_conversion_rate)
      energy_loss_to_target = energy_transfer / partial_energy_loss_rate
      target_cell.energy -= energy_loss_to_target

    return energy_transfer

  def reproduce(self, environment: "EevEnvironment"):
    # Assuming this is the reproduction weight
    reproduction_offset_from_weight = self.action_weights[2]
    if self.energy > (energy_cost_of_reproduction - reproduction_offset_from_weight):
      child_weights = self.action_weights.copy()
      if random.random() < mutation_rate:
        child_weights += np.random.uniform(-0.05, 0.05, 4)
        child_weights /= child_weights.sum() * 4
      child_position = self.position  # Modify for different spawn location
      new_cell = Cell(action_weights=child_weights,
                      position=child_position)
      environment.add_cell(new_cell)
      self.energy -= energy_cost_of_reproduction

  def dock(self, environment: "EevEnvironment"):
    # Assuming this is the docking weight
    weight = self.action_weights[3]

    # Quadratic scaling for docking probability
    # Adjust the coefficients to fit the desired curve
    docking_probability = min(0.12 * weight ** 2, 0.97)

    if random.random() < docking_probability:
      target_cell = environment.get_cell_at_position(self.get_front_position())
      environment.union(self, target_cell)
      self.is_docked = True
      target_cell.is_docked = True

  def die(self):
    self.is_dead = True


class EevEnvironment:
  def __init__(self, size, initial_cell_count):
    self.size = size
    self.grid = [[[] for _ in range(size)]
                 for _ in range(size)]  # Initialize grid
    self.cells = [Cell(position=(random.randint(
        0, size - 1), random.randint(0, size - 1))) for _ in range(initial_cell_count)]
    for cell in self.cells:
      self.place_in_grid(cell)
    self.heat = 0
    # Initial disjoint sets
    self.cell_sets = {cell: cell for cell in self.cells}

  def find(self, cell: "Cell"):
    if cell not in self.cell_sets:
        # Handle the case where the cell is not in self.cell_sets
        # For example, you could add the cell to self.cell_sets with itself as the parent
      self.cell_sets[cell] = cell
    while self.cell_sets[cell] != cell:
      cell = self.cell_sets[cell]
    return cell

  def union(self, cell1, cell2):
    # Merge the sets that cell1 and cell2 belong to
    root1 = self.find(cell1)
    root2 = self.find(cell2)
    if root1 != root2:
      self.cell_sets[root1] = root2  # Merge the sets

  def get_organism_cells(self, organism_root):
    # Return all cells that are part of the organism represented by organism_root
    return [cell for cell in self.cells if self.find(cell) == organism_root]

  def place_in_grid(self, cell: Cell):
    x, y = cell.position
    self.grid[x][y].append(cell)

  def get_cell_at_position(self, position) -> Cell | None:
    x, y = position
    # Check if the position is within the bounds of the environment
    if 0 <= x < self.size and 0 <= y < self.size:
      # If there are any cells at this grid position, return one of them
      if self.grid[x][y]:
        # Return the first cell found at this position
        return self.grid[x][y][0]
    return None  # No cell found at this position or position is out of bounds

  def update_grid(self, cell: Cell, old_position):
    old_x, old_y = old_position
    new_x, new_y = cell.position

    # Remove cell from old position
    if cell in self.grid[old_x][old_y]:
      self.grid[old_x][old_y].remove(cell)
    else:
      raise Exception("Cell not found in grid")

    # Add cell to new position
    self.grid[new_x][new_y].append(cell)

  def check_ray(self, start_position, direction, max_distance=3):
    x, y = start_position
    for _ in range(max_distance):
      # Update x, y based on direction
      if direction == "up":
        y -= 1
      elif direction == "down":
        y += 1
      elif direction == "left":
        x -= 1
      elif direction == "right":
        x += 1

      # Check if the new position is valid
      if not (0 <= x < self.size and 0 <= y < self.size):
        return False  # Ray is out of bounds

      # Check if there is a cell at the new position
      if self.check_position((x, y)):
        return True  # Cell or object found

    return False  # No cell or object found within the ray's path

  def check_position(self, position):
    x, y = position
    # Check if the position is within the bounds of the environment
    if 0 <= x < self.size and 0 <= y < self.size:
      # Return True if there are any cells at this grid position
      return bool(self.grid[x][y])
    else:
      # Position is out of bounds
      return False

  def step(self):
    for cell in self.cells:
      cell.act(self)
      self.apply_rules(cell)

    self.remove_dead_cells()

    if self.has_enough_energy():
      self.add_cell(Cell(position=(random.randint(
          0, self.size - 1), random.randint(0, self.size - 1))))

  def add_cell(self, cell: Cell):
    # Add the cell to the cells list
    self.cells.append(cell)
    # Place the cell in the grid
    self.place_in_grid(cell)
    # Add the cell to the cell_sets for Union-Find structure
    self.cell_sets[cell] = cell

  def apply_rules(self, cell: Cell):
    cell.size -= cell.decay_rate
    self.heat += cell.decay_rate
    if cell.size <= 0:
      cell.is_dead = True

  def remove_dead_cells(self):
    self.cells = [cell for cell in self.cells if not cell.is_dead]

  def remove_cell(self, cell: Cell):
    # Remove the cell from the grid
    x, y = cell.position
    if cell in self.grid[x][y]:
      self.grid[x][y].remove(cell)

    # Remove the cell from the list of cells
    if cell in self.cells:
      self.cells.remove(cell)

  def has_enough_energy(self):
    # Logic to determine if the environment can support a new cell
    return random.choice([True, False])  # Example condition


def run():

  env = EevEnvironment(size=environment_size,
                       initial_cell_count=starting_cell_count)
  for _ in range(100):
    clear_screen()
    for row in env.grid:
      row_display = ""
      for cell in row:
        if cell:  # If the list at this grid position is not empty
          row_display += "O "
        else:
          row_display += ". "  # Using a dot as a placeholder for empty positions
      print(row_display)
    env.step()


if __name__ == "__main__":
  run()  # Run the example
