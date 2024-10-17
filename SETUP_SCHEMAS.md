# Setting Up Schemas in the Dummy Data Generation Library

## Overview

Schemas define the structure and properties of the data you want to generate using the Dummy library. Each column in a schema can be customized with various attributes, including data type, nullability, duplicates, and randomizers.

## Schema Structure

A schema is defined as a dictionary where each key is a column name, and its value is another dictionary specifying the column's properties. Below are the key properties you can set for each column:

### Properties

- **type**: Specifies the data type of the column. Common types include:
  - `int`: Integer values
  - `float`: Floating-point values
  - `str`: String values
- **nullable**: A float value between `0.0` and `1.0` indicating the probability of a column containing null values.
  - `0.0`: No null values
  - `1.0`: All values are null
- **allow_duplicates**: A boolean value indicating whether duplicate values are allowed in the column.
  - `True`: Duplicates are allowed
  - `False`: No duplicates allowed
- **randomizer**: A callable function or a fixed list used to generate data for the column. If set to `None`, a default randomizer will be used based on the column type.

### Example Schema

Here’s a simple example of a schema for generating a dataset with three columns: `id`, `name`, and `score`.

```python
schema = {
    'id': {
        'type': 'int',
        'nullable': 0.0,
        'allow_duplicates': False,
        'randomizer': None  # Use default randomizer for integers
    },
    'name': {
        'type': 'str',
        'nullable': 0.1,  # 10% chance of null values
        'allow_duplicates': True,
        'randomizer': ['Alice', 'Bob', 'Charlie']  # Fixed list randomizer
    },
    'score': {
        'type': 'float',
        'nullable': 0.2,  # 20% chance of null values
        'allow_duplicates': False,
        'randomizer': None  # Use default randomizer for floats
    }
}
```

## Detailed Property Descriptions

### 1. **Type**
Defines the data type of the column. Here are the types you can use:

- `int`: Generates random integers.
- `float`: Generates random floating-point numbers.
- `str`: Generates random strings (can be fixed values or generated dynamically).

### 2. **Nullable**
Determines the likelihood of a value being `None`. For example:

- If set to `0.0`, all values will be present.
- If set to `0.5`, approximately 50% of the values may be `None`.

### 3. **Allow Duplicates**
Controls whether duplicate values can occur in the column. For example:

- Setting this to `False` ensures all generated values are unique (excluding null values).
- Setting it to `True` allows for repeated values.

### 4. **Randomizer**
Specifies how values for the column are generated:

- **Using a Callable**: You can define a function that generates values.
  
  ```python
  import random

  def generate_random_age():
      return random.randint(18, 60)

  schema = {
      'age': {
          'type': 'int',
          'nullable': 0.0,
          'allow_duplicates': True,
          'randomizer': generate_random_age  # Custom randomizer function
      }
  }
  ```

- **Using a List**: You can provide a list of fixed values from which to randomly select.
  
  ```python
  schema = {
      'status': {
          'type': 'str',
          'nullable': 0.0,
          'allow_duplicates': False,
          'randomizer': ['Active', 'Inactive', 'Pending']  # Fixed list randomizer
      }
  }
  ```

- **Using a Lambda Function**: You can use a lambda function for simple randomizers. This is useful for generating values inline without defining a separate function.

  ```python
  schema = {
      'height': {
          'type': 'float',
          'nullable': 0.0,
          'allow_duplicates': True,
          'randomizer': lambda: random.uniform(150.0, 200.0)  # Random float between 150 and 200
      },
      'age': {
          'type': 'int',
          'nullable': 0.0,
          'allow_duplicates': True,
          'randomizer': lambda: random.randint(18, 60)  # Random integer between 18 and 60
      }
  }
  ```

## Conclusion

Setting up schemas in the Dummy library allows you to customize the data generation process to fit your needs. By specifying column types, nullability, duplicates, and randomizers, you can create realistic datasets for testing and development.

For more examples and usage, refer to the library documentation or source code.
