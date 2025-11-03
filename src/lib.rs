use pyo3::prelude::*;
use rayon::prelude::*;
use rust_stemmers::{Algorithm, Stemmer};
use std::sync::OnceLock;
use dashmap::DashMap;
use std::sync::Arc;

// Use DashMap for lock-free reads and sharded writes
type CacheKey = (u8, String);
type SharedCache = Arc<DashMap<CacheKey, String>>;

static STEM_CACHE: OnceLock<SharedCache> = OnceLock::new();

// Initialize cache with optimal capacity
fn get_cache() -> &'static SharedCache {
    STEM_CACHE.get_or_init(|| {
        Arc::new(DashMap::with_capacity_and_shard_amount(100_000, 32))
    })
}

// Convert Algorithm to u8 discriminant (compile-time optimized)
const fn algorithm_to_u8(algorithm: Algorithm) -> u8 {
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

// Optimized stemmer with thread-local caching
#[pyclass]
pub struct SnowballStemmer {
    algorithm: Algorithm,
    use_cache: bool,
}

#[pymethods]
impl SnowballStemmer {
    #[new]
    #[pyo3(signature = (lang, cache = true))]
    fn new(lang: &str, cache: bool) -> PyResult<Self> {
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
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported language: {}", lang)
            )),
        };
        Ok(SnowballStemmer { algorithm, use_cache: cache })
    }

    #[inline(always)]
    fn stem_word(&self, input: &str) -> String {
        if !self.use_cache {
            return Stemmer::create(self.algorithm).stem(input).into_owned();
        }

        let cache_key = (algorithm_to_u8(self.algorithm), input.to_string());
        
        // Fast path: try to read from cache without cloning
        if let Some(entry) = get_cache().get(&cache_key) {
            return entry.value().clone();
        }
        
        // Cache miss: stem and cache
        let result = Stemmer::create(self.algorithm).stem(input).into_owned();
        get_cache().insert(cache_key, result.clone());
        result
    }

    #[inline(always)]
    pub fn stem_words_parallel(&self, py: Python<'_>, inputs: Vec<String>) -> PyResult<Vec<String>> {
        let algorithm = self.algorithm;
        let use_cache = self.use_cache;
        
        let result = py.detach(|| {
            if !use_cache {
                // Fast path without cache
                return inputs
                    .par_iter()
                    .with_min_len(500) // Increased chunk size for better throughput
                    .map(|word| {
                        Stemmer::create(algorithm).stem(word).into_owned()
                    })
                    .collect::<Vec<String>>();
            }

            // Cache-enabled path with batching
            let cache = get_cache();
            let algorithm_discriminant = algorithm_to_u8(algorithm);
            
            inputs
                .par_iter()
                .with_min_len(250) // Optimal for cache-heavy workload
                .map(|word| {
                    let cache_key = (algorithm_discriminant, word.clone());
                    
                    // Try read-only lookup first
                    if let Some(entry) = cache.get(&cache_key) {
                        return entry.value().clone();
                    }
                    
                    // Cache miss: compute and store
                    let result = Stemmer::create(algorithm).stem(word).into_owned();
                    cache.insert(cache_key, result.clone());
                    result
                })
                .collect::<Vec<String>>()
        });
        Ok(result)
    }

    #[inline(always)]
    pub fn stem_words(&self, inputs: Vec<String>) -> Vec<String> {
        if !self.use_cache {
            return inputs
                .iter()
                .map(|word| {
                    Stemmer::create(self.algorithm).stem(word).into_owned()
                })
                .collect();
        }

        let cache = get_cache();
        let algorithm_discriminant = algorithm_to_u8(self.algorithm);
        
        inputs
            .iter()
            .map(|word| {
                let cache_key = (algorithm_discriminant, word.clone());
                
                if let Some(entry) = cache.get(&cache_key) {
                    return entry.value().clone();
                }
                
                let result = Stemmer::create(self.algorithm).stem(word).into_owned();
                cache.insert(cache_key, result.clone());
                result
            })
            .collect()
    }
}

#[pymodule]
fn py_rust_stemmers(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnowballStemmer>()?;
    Ok(())
}