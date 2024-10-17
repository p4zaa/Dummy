import unittest
import random
import pandas as pd
import polars as pl
from Dummy import Dummy
import numpy as np


class TestDummy(unittest.TestCase):

    def setUp(self):
        # Basic schema setup for testing
        self.schemas = {
            'id': {
                'type': 'int',
                'nullable': 0.0,  # No null values
                'allow_duplicates': False,
                'randomizer': None  # Use default randomizer for int
            },
            'name': {
                'type': 'str',
                'nullable': 0.1,  # 10% chance of null values
                'allow_duplicates': True,
                'randomizer': None  # Use default randomizer for str
            },
            'age': {
                'type': 'int',
                'nullable': 0.0,  # No null values
                'allow_duplicates': True,
                'randomizer': lambda : random.Random(42).randint(18, 60)  # Custom randomizer for int
            },
            'score': {
                'type': 'float',
                'nullable': 0.2,  # 20% chance of null values
                'allow_duplicates': False,
                'randomizer': None  # Use default randomizer for float
            },
            'fixed_list_column': {
                'type': 'str',
                'nullable': 0.0,  # No null values
                'allow_duplicates': False,
                'randomizer': [f'A{n}' for n in range(100)]  # Fixed list randomizer
            }
        }
        self.n_samples = 100

    def test_default_randomizer(self):
        """Test generation using default randomizers."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()
        
        # Check if data has the expected number of samples
        self.assertEqual(len(data['id']), self.n_samples)
        self.assertEqual(len(data['name']), self.n_samples)

        # Check if the default randomizer generated integers
        id_sample = [x for x in data['id'] if x is not None][0]
        self.assertIsInstance(id_sample, int)
        self.assertTrue(1 <= id_sample <= 99999)

        # Check for string randomizer
        name_sample = [x for x in data['name'] if x is not None][0]
        self.assertIsInstance(name_sample, str)

    def test_no_duplicates(self):
        """Test if no duplicates are generated for columns where duplicates are not allowed, excluding None values."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Ensure no duplicates in 'id' and 'score', ignoring None values
        non_none_ids = [x for x in data['id'] if x is not None]
        non_none_scores = [x for x in data['score'] if x is not None]

        # Check that the number of unique values in non-None data equals the total non-None data
        self.assertEqual(len(non_none_ids), len(set(non_none_ids)), "Duplicates found in 'id' column")
        self.assertEqual(len(non_none_scores), len(set(non_none_scores)), "Duplicates found in 'score' column")

    def test_nullable(self):
        """Test if nullable columns have null values based on probability."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Check that there are some None values in 'name' and 'score'
        null_name_count = data['name'].count(None)
        null_score_count = data['score'].count(None)

        self.assertGreaterEqual(null_name_count, 1)
        self.assertGreaterEqual(null_score_count, 1)

    def test_fixed_list_randomizer(self):
        """Test randomizer based on a fixed list."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Ensure all values in 'fixed_list_column' are from the provided list
        for value in data['fixed_list_column']:
            self.assertIn(value, [f'A{n}' for n in range(100)])

        # Ensure no duplicates in 'fixed_list_column'
        self.assertEqual(len(data['fixed_list_column']), len(set(data['fixed_list_column'])))

    def test_polars_dataframe(self):
        """Test conversion to Polars DataFrame."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, polars=True, seed=42)
        df = dummy.to_dataframe()

        # Ensure it returns a Polars DataFrame
        self.assertIsInstance(df, pl.DataFrame)

        # Check if the columns exist
        self.assertIn('id', df.columns)
        self.assertIn('name', df.columns)

    def test_pandas_dataframe(self):
        """Test conversion to Pandas DataFrame."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, polars=False, seed=42)
        df = dummy.to_dataframe()

        # Ensure it returns a Pandas DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Check if the columns exist
        self.assertIn('id', df.columns)
        self.assertIn('name', df.columns)

    def test_high_null_prob(self):
        """Test if a high null probability generates mostly None values."""
        high_null_schema = {
            'high_null_col': {
                'type': 'str',
                'nullable': 0.9,  # 90% chance of null values
                'allow_duplicates': True
            }
        }
        dummy = Dummy(high_null_schema, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Check if most of the values are None
        null_count = data['high_null_col'].count(None)
        self.assertGreaterEqual(null_count, int(self.n_samples * 0.9), "Not enough None values generated")

    def test_custom_randomizer(self):
        """Test if custom randomizer generates data as expected."""
        custom_schemas = {
            'custom_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': lambda: "custom_value"
            }
        }
        dummy = Dummy(custom_schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Check if all values in 'custom_col' are the custom randomizer value
        self.assertTrue(all(x == "custom_value" for x in data['custom_col']))

    def test_empty_dataframe(self):
        """Test that an empty DataFrame is returned when n_samples is 0."""
        dummy = Dummy(self.schemas, n_samples=0, seed=42)
        data = dummy.dummy()

        # Ensure the generated data has no samples
        self.assertEqual(len(data['id']), 0)
        self.assertEqual(len(data['name']), 0)

        # Ensure the DataFrame has no rows
        df = dummy.to_dataframe()
        self.assertEqual(df.shape[0], 0)

    def test_mixed_types(self):
        """Test generation with mixed types (int, float, and str)."""
        dummy = Dummy(self.schemas, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Ensure each column has the expected data type
        self.assertIsInstance(data['id'][0], (int, type(None)))
        self.assertIsInstance(data['score'][0], (float, type(None)))
        self.assertIsInstance(data['name'][0], (str, type(None)))

    def test_seed_consistency(self):
        """Test that setting the same seed results in consistent data generation."""
        dummy1 = Dummy(self.schemas, n_samples=self.n_samples, seed=888)
        dummy2 = Dummy(self.schemas, n_samples=self.n_samples, seed=888)

        data1 = dummy1.dummy()
        data2 = dummy2.dummy()

        # Check that the generated data is the same for both instances
        for col in data1.keys():
            self.assertEqual(data1[col], data2[col], f"Data mismatch in column: {col}")

    def test_empty_schema(self):
        """Test that an empty schema returns an empty DataFrame."""
        dummy = Dummy({}, n_samples=self.n_samples, seed=42)
        data = dummy.dummy()

        # Ensure that no columns are generated
        self.assertEqual(len(data), 0)

        # Ensure the DataFrame has no columns
        df = dummy.to_dataframe()
        self.assertEqual(len(df.columns), 0)

    def test_fixed_list_no_duplicates(self):
        """Test that a fixed list randomizer generates no duplicates when allow_duplicates is False."""
        dummy = Dummy(self.schemas, n_samples=50, seed=42)
        data = dummy.dummy()

        # Ensure no duplicates in 'fixed_list_column'
        self.assertEqual(len(data['fixed_list_column']), len(set(data['fixed_list_column'])), "Duplicates found in fixed list column")

    def test_list_randomizer_with_duplicates(self):
        """Test that a list randomizer allows duplicates when allow_duplicates is True."""
        list_schema = {
            'list_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['apple', 'banana', 'cherry']  # List as randomizer
            }
        }
        dummy = Dummy(list_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure values are from the list and duplicates are allowed
        self.assertTrue(all(x in ['apple', 'banana', 'cherry'] for x in data['list_col']))
        self.assertGreater(len(data['list_col']), len(set(data['list_col'])), "Duplicates should be allowed")

    def test_list_randomizer_no_duplicates(self):
        """Test that a list randomizer does not allow duplicates when allow_duplicates is False."""
        list_schema = {
            'list_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': False,
                'randomizer': ['apple', 'banana', 'cherry']  # List as randomizer
            }
        }
        dummy = Dummy(list_schema, n_samples=3, seed=42)
        data = dummy.dummy()

        # Ensure values are from the list and there are no duplicates
        self.assertTrue(all(x in ['apple', 'banana', 'cherry'] for x in data['list_col']))
        self.assertEqual(len(data['list_col']), len(set(data['list_col'])), "Duplicates should not be allowed")

    def test_list_randomizer_insufficient_values(self):
        """Test that an error is raised if the list randomizer has insufficient values and duplicates are not allowed."""
        list_schema = {
            'list_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': False,
                'randomizer': ['apple', 'banana']  # Fewer values than n_samples
            }
        }
        
        with self.assertRaises(ValueError):
            Dummy(list_schema, n_samples=3, seed=42).dummy()

    def test_list_randomizer_with_nullable(self):
        """Test that null values are generated based on the null probability when using a list randomizer."""
        list_schema = {
            'list_col': {
                'type': 'str',
                'nullable': 0.5,  # 50% chance of null values
                'allow_duplicates': True,
                'randomizer': ['apple', 'banana', 'cherry']  # List as randomizer
            }
        }
        dummy = Dummy(list_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure some None values are generated and all other values come from the list
        null_count = data['list_col'].count(None)
        self.assertGreater(null_count, 0, "Expected some None values")
        self.assertTrue(all(x in ['apple', 'banana', 'cherry'] or x is None for x in data['list_col']))

    def test_none_n_samples(self):
        """Test that None for n_samples uses the length of the shortest column."""
        banks = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6']
        segments = ['s1', 's2']
        n_segment = len(segments)
        schemas = {
            'segment': {
                'type': 'str',
                'nullable': 0,
                'allow_duplicates': True,
                'randomizer': [segment for segment in segments for _ in range(len(banks))]
            },
            'bank': {
                'type': 'str',
                'nullable': 0,
                'allow_duplicates': True,
                'randomizer': banks * n_segment
            },
        }

        dummy = Dummy(schemas, n_samples=None, seed=42)
        data = dummy.dummy()

        # Check that the data length matches the expected length
        self.assertEqual(len(data['segment']), 12)
        self.assertEqual(len(data['bank']), 12)

        # Check if all values in 'segment' and 'bank' are from the provided lists
        for value in data['segment']:
            self.assertIn(value, segments)

        for value in data['bank']:
            self.assertIn(value, banks)

    def test_large_list_randomizer(self):
        """Test that a large list randomizer works as expected."""
        large_list = [f'item_{i}' for i in range(1000)]  # Large list with 1000 items
        large_list_schema = {
            'large_list_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': large_list  # Large list as randomizer
            }
        }
        dummy = Dummy(large_list_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure all values are from the provided large list
        self.assertTrue(all(x in large_list for x in data['large_list_col']))

    def test_fixed_list_with_high_nullable(self):
        """Test that a fixed list randomizer works with high nullable probability."""
        fixed_list_schema = {
            'high_null_col': {
                'type': 'str',
                'nullable': 0.8,  # 80% chance of null values
                'allow_duplicates': True,
                'randomizer': ['apple', 'banana', 'cherry']  # Fixed list
            }
        }
        dummy = Dummy(fixed_list_schema, n_samples=100, seed=None)
        data = dummy.dummy()

        # Check for None values
        null_count = data['high_null_col'].count(None)
        self.assertGreaterEqual(null_count, 70, "Expected at least 80 None values")
        
        # Ensure values are from the list where not None
        for value in data['high_null_col']:
            if value is not None:
                self.assertIn(value, ['apple', 'banana', 'cherry'])

    def test_randomizer_with_empty_list(self):
        """Test that an error is raised when using an empty list as a randomizer."""
        empty_list_schema = {
            'empty_list_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': False,
                'randomizer': []  # Empty list
            }
        }
        
        with self.assertRaises(ValueError):
            Dummy(empty_list_schema, n_samples=10, seed=42).dummy()

    def test_multiple_randomizers(self):
        """Test that columns can use different types of randomizers in one schema."""
        multiple_randomizers_schema = {
            'int_col': {
                'type': 'int',
                'nullable': 0.0,
                'allow_duplicates': False,
                'randomizer': None  # Use default
            },
            'string_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['cat', 'dog', 'fish']
            }
        }
        dummy = Dummy(multiple_randomizers_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure no duplicates in int_col and all values in string_col are from the list
        self.assertEqual(len(data['int_col']), len(set(data['int_col'])), "Duplicates found in 'int_col'")
        self.assertTrue(all(x in ['cat', 'dog', 'fish'] for x in data['string_col']))

    def test_randomizer_for_multiple_columns(self):
        """Test that the same list randomizer can be used for multiple columns."""
        shared_randomizer_schema = {
            'first_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['foo', 'bar', 'baz']  # Shared list
            },
            'second_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['foo', 'bar', 'baz']  # Shared list
            }
        }
        dummy = Dummy(shared_randomizer_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure values are from the shared list
        for col in ['first_col', 'second_col']:
            self.assertTrue(all(x in ['foo', 'bar', 'baz'] for x in data[col]))

        # Check for duplicates in both columns
        for col in ['first_col', 'second_col']:
            self.assertGreater(len(data[col]), len(set(data[col])), f"Duplicates not allowed in {col}")

    def test_minimum_samples(self):
        """Test that the minimum sample of 1 is handled correctly."""
        single_sample_schema = {
            'sample_col': {
                'type': 'int',
                'nullable': 0.0,
                'allow_duplicates': False,
                'randomizer': None  # Default randomizer
            }
        }
        dummy = Dummy(single_sample_schema, n_samples=1, seed=42)
        data = dummy.dummy()

        # Ensure one sample is generated
        self.assertEqual(len(data['sample_col']), 1)

    def test_randomizer_with_non_callable(self):
        """Test that a non-callable randomizer raises an error."""
        non_callable_schema = {
            'non_callable_col': {
                'type': 'int',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': 42  # Invalid randomizer
            }
        }
        
        with self.assertRaises(TypeError):
            Dummy(non_callable_schema, n_samples=10, seed=42).dummy()

    def test_high_duplicates(self):
        """Test that allowing duplicates generates them as expected."""
        high_duplicates_schema = {
            'dup_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['alpha', 'beta', 'gamma']  # Small list
            }
        }
        dummy = Dummy(high_duplicates_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Check for the presence of duplicates
        self.assertGreater(len(data['dup_col']), len(set(data['dup_col'])), "No duplicates found despite allowance.")

    def test_empty_nullable_schema(self):
        """Test a schema with nullable columns but no valid randomizer."""
        empty_nullable_schema = {
            'nullable_col': {
                'type': 'str',
                'nullable': 1.0,  # 100% chance of null values
                'allow_duplicates': True,
                'randomizer': None  # No randomizer
            }
        }
        dummy = Dummy(empty_nullable_schema, n_samples=10, seed=42)
        data = dummy.dummy()

        # Ensure all values are None
        self.assertTrue(all(value is None for value in data['nullable_col']))

    def test_zero_nullable_schema(self):
        """Test a schema with 0% nullable chance."""
        zero_nullable_schema = {
            'no_null_col': {
                'type': 'str',
                'nullable': 0.0,  # 0% chance of null values
                'allow_duplicates': True,
                'randomizer': ['x', 'y', 'z']  # Fixed list
            }
        }
        dummy = Dummy(zero_nullable_schema, n_samples=10, seed=42)
        data = dummy.dummy()

        # Ensure there are no None values
        self.assertFalse(any(value is None for value in data['no_null_col']))

    def test_schema_with_no_types(self):
        """Test a schema with columns defined but no types set."""
        no_type_schema = {
            'undefined_col': {
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['one', 'two', 'three']  # Valid randomizer
            }
        }

        with self.assertRaises(KeyError):
            Dummy(no_type_schema, n_samples=10, seed=42).dummy()

    def test_schema_with_empty_strings(self):
        """Test a schema that generates empty strings for string columns."""
        empty_string_schema = {
            'empty_string_col': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['']  # List with an empty string
            }
        }
        dummy = Dummy(empty_string_schema, n_samples=10, seed=42)
        data = dummy.dummy()

        # Ensure all values are empty strings
        self.assertTrue(all(value == '' for value in data['empty_string_col']))

    def test_schema_with_large_n_samples(self):
        """Test performance and correctness with a very large number of samples."""
        large_sample_schema = {
            'large_col': {
                'type': 'int',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': None  # Default randomizer
            }
        }
        dummy = Dummy(large_sample_schema, n_samples=1000000, seed=42)
        data = dummy.dummy()

        # Ensure the number of samples matches
        self.assertEqual(len(data['large_col']), 1000000)

    def test_duplicate_values_with_fixed_list(self):
        """Test that fixed list randomizer generates duplicates when allowed."""
        fixed_list_schema = {
            'fixed_list_dup': {
                'type': 'str',
                'nullable': 0.0,
                'allow_duplicates': True,
                'randomizer': ['x', 'y', 'z']  # Fixed list with allowance for duplicates
            }
        }
        dummy = Dummy(fixed_list_schema, n_samples=100, seed=42)
        data = dummy.dummy()

        # Ensure duplicates can occur
        self.assertGreater(len(data['fixed_list_dup']), len(set(data['fixed_list_dup'])), "Duplicates should occur but did not.")

if __name__ == '__main__':
    unittest.main()