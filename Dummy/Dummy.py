import random
import pandas as pd
import polars as pl
import string
from typing import Callable, List, Optional, Union, Any


class Dummy:
    def __init__(self, schemas: dict, n_samples: int = 100, polars: bool = True, default_string_length: int = 5, seed: Optional[int] = None):
        self.schemas = schemas
        self.n_samples = n_samples
        self.polars = polars
        self.default_string_length = default_string_length
        self.seed = seed
        
        # Set the seed if provided
        #if self.seed is not None:
        #    random.seed(self.seed)

        self.random_gen = random.Random(seed)  # Use random.Random for local state
    
    def _default_randomizer(self, col_type: str):
        """Return a default randomizer based on the column type."""
        if col_type == 'int':
            return lambda: self.random_gen.randint(1, 99999)
        elif col_type == 'float':
            return lambda: self.random_gen.uniform(0.0, 99999.9)
        elif col_type == 'str':
            return lambda: ''.join(self.random_gen.choices(string.ascii_letters, k=self.default_string_length))
        else:
            raise ValueError(f"Unsupported type: {col_type}")

    def _randomize_column(self, config):
        """Randomize data for a single column based on its configuration."""
        col_type = config['type']
        null_prob = config.get('nullable', 0.0)
        allow_duplicates = config.get('allow_duplicates', True)
        default_randomizer = False if config.get('randomizer', None) else True

        # Set randomizer if not provided
        randomizer = config.get('randomizer', None) or self._default_randomizer(col_type)

        data = []
        unique_values = set()

        # If a fixed list of values is provided
        if isinstance(randomizer, list):
            if not allow_duplicates and len(randomizer) < self.n_samples:
                raise ValueError(f"Not enough unique values in the provided list for '{col_type}' column.")
            
            for _ in range(self.n_samples):
                if self.random_gen.random() < null_prob:
                    data.append(None)
                else:
                    if allow_duplicates:
                        data.append(self.random_gen.choice(randomizer))
                    else:
                        value = self.random_gen.choice(randomizer)
                        while value in unique_values:
                            value = self.random_gen.choice(randomizer)
                        data.append(value)
                        unique_values.add(value)

        # If no fixed list but allow_duplicates is False and no custom randomizer, generate unique values
        elif not allow_duplicates and default_randomizer:
            if col_type == 'int':
                # Generate enough unique integer values
                possible_values = range(1, 99999 + 1)
                unique_data = self.random_gen.sample(possible_values, self.n_samples)
            elif col_type == 'float':
                # Generate enough unique float values
                possible_values = [round(x * 0.1, 1) for x in range(0, 999999)]
                unique_data = self.random_gen.sample(possible_values, self.n_samples)
            elif col_type == 'str':
                # Ensure enough unique strings
                possible_values = set(randomizer() for _ in range(self.n_samples * 2))
                if len(possible_values) < self.n_samples:
                    raise ValueError(f"Not enough unique values for '{col_type}' column.")
                unique_data = self.random_gen.sample(possible_values, self.n_samples)
            else:
                raise ValueError(f"Unsupported type: {col_type}")
            
            for value in unique_data:
                if self.random_gen.random() < null_prob:
                    data.append(None)
                else:
                    data.append(value)

        # Custom randomizer or allow_duplicates case
        else:
            for _ in range(self.n_samples):
                if self.random_gen.random() < null_prob:
                    data.append(None)
                else:
                    value = randomizer()
                    if allow_duplicates or value not in unique_values:
                        data.append(value)
                        unique_values.add(value)

            # Ensure uniqueness if allow_duplicates is False
            if not allow_duplicates and len(unique_values) < self.n_samples:
                raise ValueError(f"Random generation did not provide enough unique values for '{col_type}' column.")

        return data

    def _generate_data(self):
        """Generate data for all columns."""
        data = {}
        for col_name, config in self.schemas.items():
            data[col_name] = self._randomize_column(config)
        return data

    def dummy(self):
        """Return the raw generated data as a dictionary."""
        return self._generate_data()

    def to_dataframe(self):
        """Convert generated data to either Polars or Pandas DataFrame."""
        data = self._generate_data()

        if self.polars:
            return pl.DataFrame(data)
        else:
            return pd.DataFrame(data)