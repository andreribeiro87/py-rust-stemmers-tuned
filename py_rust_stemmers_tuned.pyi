"""Type stubs for py_rust_stemmers_tuned - High-performance Snowball stemmer implementation."""

from typing import List

class SnowballStemmer:
    """
    High-performance Snowball stemmer with optional caching.
    
    Supports the following languages:
    - arabic, danish, dutch, english, finnish, french, german, greek,
    - hungarian, italian, norwegian, portuguese, romanian, russian,
    - spanish, swedish, tamil, turkish
    
    Args:
        lang: Language name (case-insensitive)
        cache: Whether to use caching for better performance with repeated words (default: True)
    
    Raises:
        ValueError: If the language is not supported
    """
    
    def __init__(self, lang: str, cache: bool = True) -> None:
        """
        Initialize a Snowball stemmer for the specified language.
        
        Args:
            lang: Language name (case-insensitive)
            cache: Enable caching for repeated words (default: True)
        
        Raises:
            ValueError: If the language is not supported
        """
        ...
    
    def stem_word(self, input: str) -> str:
        """
        Stem a single word.
        
        Args:
            input: The word to stem
        
        Returns:
            The stemmed word
        """
        ...
    
    def stem_words(self, inputs: List[str]) -> List[str]:
        """
        Stem a list of words sequentially.
        
        This method processes words one by one and is suitable for smaller batches.
        For large batches, consider using stem_words_parallel for better performance.
        
        Args:
            inputs: List of words to stem
        
        Returns:
            List of stemmed words in the same order as input
        """
        ...
    
    def stem_words_parallel(self, inputs: List[str]) -> List[str]:
        """
        Stem a list of words in parallel using multiple threads.
        
        This method is optimized for processing large batches of words.
        It automatically determines optimal parallelism based on input size.
        
        Args:
            inputs: List of words to stem
        
        Returns:
            List of stemmed words in the same order as input
        """
        ...
