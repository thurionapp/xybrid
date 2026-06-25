//! Embedding smoke test for the llama.cpp backend.
//!
//! Validates the self-contained embedding path: `LlamaCppBackend::embed`
//! tokenizes the input, runs it through an internally-created embedding-mode
//! context, and returns a pooled vector of length `n_embd`.
//!
//! The test is gated on a real GGUF model supplied via the
//! `XYBRID_EMBED_TEST_GGUF` environment variable and skips cleanly when it is
//! unset, so CI without the asset stays green.
//!
//! Run with:
//!   XYBRID_EMBED_TEST_GGUF=/path/to/model.gguf \
//!     cargo test -p xybrid-core --test llama_embeddings_smoke --features llm-llamacpp

#![cfg(feature = "llm-llamacpp")]

use xybrid_core::runtime_adapter::llama_cpp::LlamaCppBackend;
use xybrid_core::runtime_adapter::llm::{LlmBackend, LlmConfig};

/// Cosine similarity between two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[test]
fn embed_produces_stable_semantic_vectors() {
    let Ok(model_path) = std::env::var("XYBRID_EMBED_TEST_GGUF") else {
        eprintln!("Skipping embed smoke: set XYBRID_EMBED_TEST_GGUF=/path/to/model.gguf to run it");
        return;
    };

    let mut backend = LlamaCppBackend::new().expect("construct llama.cpp backend");
    backend
        .load(&LlmConfig::new(model_path))
        .expect("load GGUF model");

    assert!(
        backend.supports_embeddings(),
        "llama.cpp backend should advertise embedding support"
    );

    // MEAN pooling (1) over a couple of related and one unrelated sentence.
    let v_cat = backend
        .embed("the cat sat on the mat", 1)
        .expect("embed cat");
    let v_kitten = backend
        .embed("a kitten rested on the rug", 1)
        .expect("embed kitten");
    let v_finance = backend
        .embed("quarterly revenue exceeded forecasts", 1)
        .expect("embed finance");

    // Non-trivial, equal-length vectors.
    assert!(!v_cat.is_empty(), "embedding must be non-empty");
    assert_eq!(v_cat.len(), v_kitten.len());
    assert_eq!(v_cat.len(), v_finance.len());
    assert!(
        v_cat.iter().any(|&x| x != 0.0),
        "embedding must not be all zeros"
    );

    // Determinism: same text, same vector.
    let v_cat2 = backend
        .embed("the cat sat on the mat", 1)
        .expect("re-embed cat");
    let self_sim = cosine(&v_cat, &v_cat2);
    assert!(
        self_sim > 0.999,
        "same text should embed deterministically (cos={self_sim})"
    );

    // Semantics: the two cat/kitten sentences should be closer to each other
    // than either is to the finance sentence. This is a soft sanity check —
    // generative GGUFs are not trained as embedders, so we only require the
    // related pair to out-rank the unrelated one.
    let related = cosine(&v_cat, &v_kitten);
    let unrelated = cosine(&v_cat, &v_finance);
    println!("cos(cat,kitten)={related}  cos(cat,finance)={unrelated}");
    assert!(
        related >= unrelated,
        "related sentences should not be less similar than unrelated ones \
         (related={related}, unrelated={unrelated})"
    );
}
