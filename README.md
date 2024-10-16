
# Dummy Data Generator

`Dummy` is a Python library for generating dummy data for testing and development purposes. It provides flexibility in defining schemas for various data types, including integers, floats, and strings, with options for null values and duplicate handling.

## Features

- **Flexible Schema Definition**: Define data schemas with various types and properties.
- **Custom Randomizers**: Use built-in randomizers or define custom randomizers for specific data generation needs.
- **Nullability and Duplicates**: Control whether columns can contain null values and whether duplicates are allowed.
- **Supports Pandas and Polars**: Easily convert generated data into Pandas or Polars DataFrames for further analysis.

## Installation

To install the library, clone the repository and use the following command:

```bash
pip install -e .
```

## Usage

### Basic Example

Here is a basic example of how to use the `Dummy` library to generate dummy data:

```python
from Dummy import Dummy

# Define your schema
schemas = {
    'id': {
        'type': 'int',
        'nullable': 0.0,
        'allow_duplicates': False,
    },
    'name': {
        'type': 'str',
        'nullable': 0.1,
        'allow_duplicates': True,
    },
    'age': {
        'type': 'int',
        'nullable': 0.0,
        'allow_duplicates': True,
        'randomizer': lambda: random.randint(18, 60),
    },
    'score': {
        'type': 'float',
        'nullable': 0.2,
        'allow_duplicates': False,
    }
}

# Create an instance of Dummy
dummy = Dummy(schemas, n_samples=100, seed=42)

# Generate dummy data
data = dummy.dummy()
print(data)
```

### Custom Randomizers

You can define custom randomizers for more specific data generation needs. For example:

```python
def custom_name_randomizer(random_gen):
    return ''.join(random_gen.choices('abcdefghijklmnopqrstuvwxyz', k=5))

schemas['name']['randomizer'] = custom_name_randomizer

# Generate dummy data with custom randomizer
data = dummy.dummy()
print(data)
```

### Convert to DataFrame

You can easily convert the generated data into a Pandas or Polars DataFrame:

```python
import pandas as pd

# Convert to Pandas DataFrame
df = dummy.to_dataframe(polars=False)
print(df)

# Convert to Polars DataFrame
import polars as pl
df_polars = dummy.to_dataframe(polars=True)
print(df_polars)
```

## Testing

The library includes a suite of unit tests to ensure functionality. To run the tests, execute:

```bash
python -m unittest discover -s tests
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
