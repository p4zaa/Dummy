
# Setting Up Schemas for Dummy Data Generation

## Overview

The `Dummy` class is designed to generate dummy data based on defined schemas. The schemas dictate the structure and characteristics of the data, including types, nullability, randomizers, and other configurations.

## Initializing the Dummy Class

To use the `Dummy` class, you need to provide schemas along with optional parameters. Here’s how to initialize it:

```python
from your_dummy_module import Dummy

schemas = {
    'column_name': {
        'type': 'int',  # or 'float', 'str'
        'nullable': 0.1,  # Probability of null values
        'randomizer': [1, 2, 3, 4, 5],  # Optional fixed list of values
        'allow_duplicates': True,  # Whether to allow duplicate values
    },
    # Add more columns as needed
}

dummy_generator = Dummy(schemas=schemas, n_samples=None, polars=True, default_string_length=5, seed=42)
```

### Parameters

- `schemas` (dict): A dictionary defining the schema for each column.
- `n_samples` (int, optional): The number of samples to generate. If not provided, it is set automatically based on the shortest list in the schema.
- `polars` (bool): If `True`, the output will be a Polars DataFrame; otherwise, it will be a Pandas DataFrame.
- `default_string_length` (int): Default length for generated strings.
- `seed` (int, optional): Seed for random number generation to ensure reproducibility.

## Defining Schemas

Each column in the schemas can have the following properties:

- `type`: The data type of the column. Supported types are `'int'`, `'float'`, and `'str'`.
- `nullable`: Probability (between 0 and 1) of generating a `None` value.
- `randomizer`: A list of values from which to randomly select, if provided. If not provided, a default randomizer is used based on the column type.
- `allow_duplicates`: Whether to allow duplicate values in the column. Defaults to `True`.

### Example of a Schema Definition

```python
schemas = {
    'age': {
        'type': 'int',
        'nullable': 0.1,
        'randomizer': list(range(18, 60)),  # Age range from 18 to 59
        'allow_duplicates': False,
    },
    'salary': {
        'type': 'float',
        'nullable': 0.2,
        'randomizer': None,  # Will use default randomizer
        'allow_duplicates': True,
    },
    'name': {
        'type': 'str',
        'nullable': 0.05,
        'randomizer': [ 'Alice', 'Bob', 'Charlie', 'David', 'Eve' ],
        'allow_duplicates': True,
    },
}
```

## Using Lambda Functions in Randomizers

You can also use lambda functions as randomizers for generating data. Here’s an example of how to use a lambda function for the `salary` column:

```python
schemas = {
    'salary': {
        'type': 'float',
        'nullable': 0.2,
        'randomizer': lambda: round(random.uniform(30000, 120000), 2),  # Salary between 30k and 120k
        'allow_duplicates': True,
    },
}
```

### Note
- Ensure that the randomizer matches the specified column type.
- If using a list as a randomizer, the number of unique values should be sufficient to meet the `n_samples` requirement if `allow_duplicates` is set to `False`.

## Conclusion

With the `Dummy` class, you can easily set up schemas for generating a variety of dummy data. This is particularly useful for testing, prototyping, or simulating data in applications. Adjust the parameters and schema definitions as needed to fit your specific use case.
