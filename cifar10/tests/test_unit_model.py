# -*- coding: utf-8 -*-
import unittest
import cifar10.models.deepaas_api as deepaas_api

class TestModelMethods(unittest.TestCase):
    
    def setUp(self):
        self.meta = deepaas_api.get_metadata()
        
    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)
        
    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['Name'].replace('-','').replace('_',''),
                        'cifar10'.replace('-','').replace('_',''))
        self.assertEqual(self.meta['Author'], 'lara')
        self.assertEqual(self.meta['Author-email'], 'lloret@ifca.unican.es')


if __name__ == '__main__':
    unittest.main()
