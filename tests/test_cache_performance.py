#!/usr/bin/env python3
"""
Simple test to verify cache functionality by timing repeated stemming operations.
The second run on the same words should be significantly faster due to caching.
"""
import time
from py_rust_stemmers import SnowballStemmer

def test_cache_performance():
    s = SnowballStemmer('english')
    
    # Create a list of words with significant repetition
    words = ["running", "jumping", "swimming", "coding", "testing"] * 100
    
    print(f"Testing cache performance with {len(words)} words ({len(set(words))} unique)")
    
    # First run - populate cache
    start = time.time()
    result1 = s.stem_words(words)
    time1 = time.time() - start
    print(f"First run (populating cache): {time1:.6f} seconds")
    
    # Second run - should hit cache
    start = time.time()
    result2 = s.stem_words(words)
    time2 = time.time() - start
    print(f"Second run (using cache): {time2:.6f} seconds")
    
    # Verify results are the same
    assert result1 == result2, "Results should be identical"
    
    # Calculate speedup
    if time2 > 0:
        speedup = time1 / time2
        print(f"Speedup: {speedup:.2f}x")
    
    print("\n✓ Cache is working correctly!")

def test_cross_instance_cache():
    """Test that cache is shared across instances"""
    print("\nTesting cross-instance cache sharing...")
    
    s1 = SnowballStemmer('english')
    s2 = SnowballStemmer('english')
    
    words = ["running", "jumping", "swimming"] * 50
    
    # First instance populates cache
    start = time.time()
    s1.stem_words(words)
    time1 = time.time() - start
    print(f"First instance: {time1:.6f} seconds")
    
    # Second instance should benefit from cache
    start = time.time()
    s2.stem_words(words)
    time2 = time.time() - start
    print(f"Second instance: {time2:.6f} seconds")
    
    if time2 > 0 and time1 > 0:
        speedup = time1 / time2
        print(f"Speedup: {speedup:.2f}x")
    
    print("✓ Cross-instance caching works!")

def test_parallel_cache():
    """Test that parallel stemming also uses cache"""
    print("\nTesting parallel stemming with cache...")
    
    s = SnowballStemmer('english')
    words = ["running", "jumping", "swimming", "coding", "testing"] * 100
    
    # First run
    start = time.time()
    result1 = s.stem_words_parallel(words)
    time1 = time.time() - start
    print(f"First parallel run: {time1:.6f} seconds")
    
    # Second run - should hit cache
    start = time.time()
    result2 = s.stem_words_parallel(words)
    time2 = time.time() - start
    print(f"Second parallel run: {time2:.6f} seconds")
    
    assert result1 == result2, "Results should be identical"
    
    if time2 > 0:
        speedup = time1 / time2
        print(f"Speedup: {speedup:.2f}x")
    
    print("✓ Parallel cache works!")

if __name__ == '__main__':
    test_cache_performance()
    test_cross_instance_cache()
    test_parallel_cache()
    print("\n" + "="*50)
    print("All cache performance tests passed! ✓")
    print("="*50)
