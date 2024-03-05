import random

decay_to_heat_conversion = 0.5
initial_energy = 10.0
energy_cost_of_action = 0.1
full_energy_conversion_rate = 0.8
partial_energy_conversion_rate = 0.5
partial_energy_loss_rate = 0.6
energy_cost_of_eating = 0.2


class EevEnvironment:
  def __init__(self, size, initial_cell_count):
    self.size = size  # Size of the environment (could be 2D or 3D)
    self.cells = [Cell()
                  for _ in range(initial_cell_count)]  # Initialize cells
    self.heat = 0  # Heat in the environment

  def step(self):
    # This method advances the environment by one time step.
    for cell in self.cells:
      cell.act(self)  # Each cell takes an action
      self.apply_rules(cell)  # Apply the environmental rules to the cell

    self.handle_cell_interactions()
    self.remove_dead_cells()

    for cell in self.cells:
      if cell.is_dead:
        self.convert_to_resource(cell)
      else:
        self.heat += cell.decay_rate * decay_to_heat_conversion

  def apply_rules(self, cell):
    # Apply environmental rules like decay
    cell.size -= cell.decay_rate
    if cell.size <= 0:
      cell.alive = False

  def handle_cell_interactions(self):
    # Handle interactions like consumption, docking, etc.
    pass  # You'll need to implement the logic here

  def remove_dead_cells(self):
    # Remove dead cells from the environment
    self.cells = [cell for cell in self.cells if cell.alive]

  def convert_to_resource(self, cell):
    # Logic to turn dead cells into consumable resources
    pass  # Implement the conversion logic here


class Cell:
  def __init__(self):
    self.size = random.uniform(0.5, 1.0)  # Initial size of the cell
    self.alive = True
    self.decay_rate = 0.01  # Rate at which the cell decays
    self.energy = initial_energy
    self.is_dead = False
    self.max_size = 10  # Example maximum size

  def choose_action(self):
    random.choice(["move", "eat", "reproduce", "dock"])

  def act(self):
    # Implement actions with energy costs
    self.energy -= energy_cost_of_action
    if self.energy <= 0:
      self.die()

    # Example action logic (to be expanded and refined)
    elif self.size >= self.max_size:
      self.handle_max_size_reached()
    else:
      action = self.choose_action()

  def eat(self, target_cell: "Cell", environment_heat: int):
    size_ratio = self.size / target_cell.size
    # Efficiency based on size ratio
    energy_transfer_efficiency = min(size_ratio, 1)
    new_environment_heat = environment_heat

    if size_ratio >= 1.5:
      # Full consumption
      energy_gain = target_cell.size * \
        full_energy_conversion_rate * energy_transfer_efficiency
      energy_loss_to_heat = target_cell.size * \
          (1 - full_energy_conversion_rate)
      self.size += energy_gain
      target_cell.die()
      new_environment_heat += energy_loss_to_heat
    else:
      # Partial consumption (nibbling)
      energy_gain = target_cell.size * \
        partial_energy_conversion_rate * energy_transfer_efficiency
      energy_loss = target_cell.size * partial_energy_loss_rate
      energy_loss_to_heat = energy_loss - energy_gain

      self.energy += energy_gain
      target_cell.size -= energy_loss
      new_environment_heat += energy_loss_to_heat

      if target_cell.size <= 0:
        target_cell.die()

    self.energy -= energy_cost_of_eating
    return new_environment_heat

  def mutate(self):
    # Simple mutation logic (to be refined)
    for i in range(len(self.action_multipliers)):
      self.action_multipliers[i] += random.uniform(-0.1, 0.1)
    # Ensure the sum of multipliers remains constant
    total = sum(self.action_multipliers)
    self.action_multipliers = [x / total * 4 for x in self.action_multipliers]

  def reproduce(self):
    # Reproduction logic including mutation
    offspring = Cell()
    offspring.action_multipliers = self.action_multipliers.copy()
    offspring.mutate()
    return offspring

  def die(self):
    self.is_dead = True
    # Convert cell into a resource in the environment

  def handle_max_size_reached(self):
    # Implement logic for when the cell reaches maximum size
    pass  # Placeholder for size handling logic


# Example usage
env = EevEnvironment(size=100, initial_cell_count=10)
for _ in range(100):  # Run for 100 time steps
  env.step()
