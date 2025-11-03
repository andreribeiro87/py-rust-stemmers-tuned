use pyo3::prelude::*;
use rayon::prelude::*;
// Import the stemmer implementation from the rust-stemmers library
extern crate rust_stemmers;
use rust_stemmers::{Algorithm, Stemmer};
use std::sync::{Mutex, OnceLock};
use lru::LruCache;
use std::num::NonZeroUsize;

// Global LRU cache for stemming results
// Key: (u8, String) - u8 represents the algorithm discriminant, String is the word
// Value: String - The stemmed result
type CacheKey = (u8, String);
type StemCache = Mutex<LruCache<CacheKey, String>>;

static STEM_CACHE: OnceLock<StemCache> = OnceLock::new();

// Convert Algorithm to a u8 discriminant for use in cache key
fn algorithm_to_u8(algorithm: Algorithm) -> u8 {
    match algorithm {
        Algorithm::Arabic => 0,
        Algorithm::Danish => 1,
        Algorithm::Dutch => 2,
        Algorithm::English => 3,
        Algorithm::Finnish => 4,
        Algorithm::French => 5,
        Algorithm::German => 6,
        Algorithm::Greek => 7,
        Algorithm::Hungarian => 8,
        Algorithm::Italian => 9,
        Algorithm::Norwegian => 10,
        Algorithm::Portuguese => 11,
        Algorithm::Romanian => 12,
        Algorithm::Russian => 13,
        Algorithm::Spanish => 14,
        Algorithm::Swedish => 15,
        Algorithm::Tamil => 16,
        Algorithm::Turkish => 17,
    }
}

// Initialize the cache with 100,000 capacity
fn get_cache() -> &'static StemCache {
    STEM_CACHE.get_or_init(|| {
        let capacity = NonZeroUsize::new(100_000).unwrap();
        Mutex::new(LruCache::new(capacity))
    })
}

// Create a Python class to expose the stemmer functionality
#[pyclass]
pub struct SnowballStemmer {
    stemmer: Stemmer,
    algorithm: Algorithm,
}

#[pymethods]
impl SnowballStemmer {
    #[new]
    fn new(lang: &str) -> PyResult<Self> {
        let algorithm = match lang.to_lowercase().as_str() {
            "arabic" => Algorithm::Arabic,
            "danish" => Algorithm::Danish,
            "dutch" => Algorithm::Dutch,
            "english" => Algorithm::English,
            "finnish" => Algorithm::Finnish,
            "french" => Algorithm::French,
            "german" => Algorithm::German,
            "greek" => Algorithm::Greek,
            "hungarian" => Algorithm::Hungarian,
            "italian" => Algorithm::Italian,
            "norwegian" => Algorithm::Norwegian,
            "portuguese" => Algorithm::Portuguese,
            "romanian" => Algorithm::Romanian,
            "russian" => Algorithm::Russian,
            "spanish" => Algorithm::Spanish,
            "swedish" => Algorithm::Swedish,
            "tamil" => Algorithm::Tamil,
            "turkish" => Algorithm::Turkish,
            // throw exception instead of crashing, preserve prior test behavior
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported language: {}", lang))),
        };
        let stemmer = Stemmer::create(algorithm);
        Ok(SnowballStemmer { stemmer, algorithm })
    }

    #[inline(always)]
    fn stem_word(&self, input: &str) -> String {
        let cache_key = (algorithm_to_u8(self.algorithm), input.to_string());
        
        // Try to get from cache first
        {
            let mut cache = get_cache().lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }
        
        // Cache miss - perform stemming
        let result = self.stemmer.stem(input).into_owned();
        
        // Store in cache
        {
            let mut cache = get_cache().lock().unwrap();
            cache.put(cache_key, result.clone());
        }
        
        result
    }

    #[inline(always)]
    pub fn stem_words_parallel(&self, py: Python<'_>, inputs: Vec<String>) -> PyResult<Vec<String>> {
        // release GIL
        py.allow_threads(|| {
            let result = inputs
                .par_iter()
                .map(|word| {
                    let cache_key = (algorithm_to_u8(self.algorithm), word.clone());
                    
                    // Try to get from cache first
                    {
                        let mut cache = get_cache().lock().unwrap();
                        if let Some(cached) = cache.get(&cache_key) {
                            return cached.clone();
                        }
                    }
                    
                    // Cache miss - perform stemming
                    let result = self.stemmer.stem(word.as_str()).into_owned();
                    
                    // Store in cache
                    {
                        let mut cache = get_cache().lock().unwrap();
                        cache.put(cache_key, result.clone());
                    }
                    
                    result
                })
                .collect();
            Ok(result)
        })
    }

    // refactor to Vec<String> based on the discussion(s) here: https://github.com/PyO3/pyo3/discussions/4830
    #[inline(always)]
    pub fn stem_words(&self, inputs: Vec<String>) -> Vec<String> {
        inputs
            .iter()
            .map(|word| {
                let cache_key = (algorithm_to_u8(self.algorithm), word.clone());
                
                // Try to get from cache first
                {
                    let mut cache = get_cache().lock().unwrap();
                    if let Some(cached) = cache.get(&cache_key) {
                        return cached.clone();
                    }
                }
                
                // Cache miss - perform stemming
                let result = self.stemmer.stem(word.as_str()).into_owned();
                
                // Store in cache
                {
                    let mut cache = get_cache().lock().unwrap();
                    cache.put(cache_key, result.clone());
                }
                
                result
            })
            .collect()
    }
}

/// This module is required for the Python interpreter to access the Rust functions.
#[pymodule]
fn py_rust_stemmers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnowballStemmer>()?;
    Ok(())
}