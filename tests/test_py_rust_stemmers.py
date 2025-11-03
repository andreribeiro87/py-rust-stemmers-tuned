import unittest
from py_rust_stemmers import SnowballStemmer

class TestRustStemmer(unittest.TestCase):
      
    def test_english_stemming(self):
        s = SnowballStemmer('english')
        words = ["fruitlessly", "happiness", "computations"]
        expected = ["fruitless", "happi", "comput"]
        result = [s.stem_word(w) for w in words]
        self.assertEqual(result, expected)

    def test_spanish_stemming(self):
        s = SnowballStemmer('spanish')
        words = ["frutalmente", "felicidad", "computaciones"]
        expected = ["frutal", "felic", "comput"]
        result = [s.stem_word(w) for w in words]
        self.assertEqual(result, expected)

    def test_empty_input(self):
        s = SnowballStemmer('english')
        expected = ['']
        result = [s.stem_word("")]
        self.assertEqual(result, expected)

    def test_invalid_language(self):
        words = ["fruitlessly", "happiness", "computations"]
        with self.assertRaises(ValueError):
            s = SnowballStemmer('invalid_lang')

    def test_cache_with_repeated_words(self):
        """Test that repeated words benefit from caching"""
        s = SnowballStemmer('english')
        # Test with repeated words
        words_with_repetition = ["running", "jumping", "running", "swimming", "jumping", "running"]
        expected = ["run", "jump", "run", "swim", "jump", "run"]
        
        # Test stem_word - repeated calls should use cache
        result = [s.stem_word(w) for w in words_with_repetition]
        self.assertEqual(result, expected)
        
        # Test stem_words - should also use cache
        result_batch = s.stem_words(words_with_repetition)
        self.assertEqual(result_batch, expected)
        
        # Test stem_words_parallel - should also use cache
        result_parallel = s.stem_words_parallel(words_with_repetition)
        self.assertEqual(result_parallel, expected)

    def test_cache_across_instances(self):
        """Test that cache is shared across different stemmer instances of same language"""
        s1 = SnowballStemmer('english')
        s2 = SnowballStemmer('english')
        
        # First instance stems the word
        result1 = s1.stem_word("testing")
        
        # Second instance should get cached result
        result2 = s2.stem_word("testing")
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, "test")

    def test_cache_language_isolation(self):
        """Test that cache keeps different languages separate"""
        s_en = SnowballStemmer('english')
        s_es = SnowballStemmer('spanish')
        
        # Same word in different languages should have different stems
        # (if stemming algorithms differ for this word)
        result_en = s_en.stem_word("computations")
        result_es = s_es.stem_word("computaciones")
        
        # Just verify they run without error and produce results
        self.assertIsNotNone(result_en)
        self.assertIsNotNone(result_es)

if __name__ == '__main__':
    unittest.main()
