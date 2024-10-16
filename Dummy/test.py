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

if __name__ == '__main__':
    unittest.main()