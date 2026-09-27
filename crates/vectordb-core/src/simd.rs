//! SIMD-accelerated distance metrics for vector similarity search.
//! Supports Apple Silicon (ARM NEON), x86_64 (AVX2/FMA), and fallback scalar.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

/// Calculate dot product of two f32 slices.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        unsafe { dot_product_avx2(a, b) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        dot_product_scalar(a, b)
    }
}

/// Calculate squared Euclidean (L2) distance of two f32 slices.
#[inline]
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match");

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_sq_neon(a, b) }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        unsafe { euclidean_sq_avx2(a, b) }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        euclidean_sq_scalar(a, b)
    }
}

/// Calculate L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(a: &[f32]) -> f32 {
    dot_product(a, a).sqrt()
}

/// Normalize vector to unit length in-place.
pub fn normalize_in_place(a: &mut [f32]) {
    let norm = l2_norm(a);
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        for val in a.iter_mut() {
            *val *= inv_norm;
        }
    }
}

/// Calculate cosine distance: 1.0 - cosine_similarity.
/// For pre-normalized vectors, this is simply 1.0 - dot_product(a, b).
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < 1e-12 {
        1.0
    } else {
        let sim = (dot / denom).clamp(-1.0, 1.0);
        1.0 - sim
    }
}

/// Calculate cosine similarity: in [-1.0, 1.0].
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    let denom = norm_a * norm_b;
    if denom < 1e-12 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Generic distance dispatcher according to `DistanceMetric`.
#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b), // lower distance = higher similarity
        DistanceMetric::Euclidean => euclidean_distance_sq(a, b).sqrt(),
    }
}

// ------------------------------------------------------------------------------------
// ARM NEON Implementations (Apple Silicon M1/M2/M3/M4 optimized)
// ------------------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks16 = len / 16;
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();

    for _ in 0..chunks16 {
        let a0 = vld1q_f32(pa);
        let b0 = vld1q_f32(pb);
        sum0 = vfmaq_f32(sum0, a0, b0);

        let a1 = vld1q_f32(pa.add(4));
        let b1 = vld1q_f32(pb.add(4));
        sum1 = vfmaq_f32(sum1, a1, b1);

        let a2 = vld1q_f32(pa.add(8));
        let b2 = vld1q_f32(pb.add(8));
        sum2 = vfmaq_f32(sum2, a2, b2);

        let a3 = vld1q_f32(pa.add(12));
        let b3 = vld1q_f32(pb.add(12));
        sum3 = vfmaq_f32(sum3, a3, b3);

        pa = pa.add(16);
        pb = pb.add(16);
    }

    let mut sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    let remainder4 = (len % 16) / 4;
    for _ in 0..remainder4 {
        let va = vld1q_f32(pa);
        let vb = vld1q_f32(pb);
        sum = vfmaq_f32(sum, va, vb);
        pa = pa.add(4);
        pb = pb.add(4);
    }

    let mut total = vaddvq_f32(sum);

    let remainder = len % 4;
    let offset = len - remainder;
    for i in 0..remainder {
        total += *a.get_unchecked(offset + i) * *b.get_unchecked(offset + i);
    }

    total
}

#[cfg(target_arch = "aarch64")]
unsafe fn euclidean_sq_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let len = a.len();
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let chunks16 = len / 16;
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();

    for _ in 0..chunks16 {
        let d0 = vsubq_f32(vld1q_f32(pa), vld1q_f32(pb));
        sum0 = vfmaq_f32(sum0, d0, d0);

        let d1 = vsubq_f32(vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
        sum1 = vfmaq_f32(sum1, d1, d1);

        let d2 = vsubq_f32(vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
        sum2 = vfmaq_f32(sum2, d2, d2);

        let d3 = vsubq_f32(vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
        sum3 = vfmaq_f32(sum3, d3, d3);

        pa = pa.add(16);
        pb = pb.add(16);
    }

    let mut sum = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));

    let remainder4 = (len % 16) / 4;
    for _ in 0..remainder4 {
        let diff = vsubq_f32(vld1q_f32(pa), vld1q_f32(pb));
        sum = vfmaq_f32(sum, diff, diff);
        pa = pa.add(4);
        pb = pb.add(4);
    }

    let mut total = vaddvq_f32(sum);

    let remainder = len % 4;
    let offset = len - remainder;
    for i in 0..remainder {
        let diff = *a.get_unchecked(offset + i) - *b.get_unchecked(offset + i);
        total += diff * diff;
    }

    total
}

// ------------------------------------------------------------------------------------
// x86_64 AVX2 Implementations
// ------------------------------------------------------------------------------------
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks8 = len / 8;
    let mut sum = _mm256_setzero_ps();
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();

    for _ in 0..chunks8 {
        let va = _mm256_loadu_ps(pa);
        let vb = _mm256_loadu_ps(pb);
        sum = _mm256_fmadd_ps(va, vb, sum);
        pa = pa.add(8);
        pb = pb.add(8);
    }

    let mut buffer = [0.0f32; 8];
    _mm256_storeu_ps(buffer.as_mut_ptr(), sum);
    let mut total: f32 = buffer.iter().sum();

    for i in (chunks8 * 8)..len {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
    }
    total
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
unsafe fn euclidean_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks8 = len / 8;
    let mut sum = _mm256_setzero_ps();
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();

    for _ in 0..chunks8 {
        let va = _mm256_loadu_ps(pa);
        let vb = _mm256_loadu_ps(pb);
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        pa = pa.add(8);
        pb = pb.add(8);
    }

    let mut buffer = [0.0f32; 8];
    _mm256_storeu_ps(buffer.as_mut_ptr(), sum);
    let mut total: f32 = buffer.iter().sum();

    for i in (chunks8 * 8)..len {
        let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
        total += diff * diff;
    }
    total
}

// ------------------------------------------------------------------------------------
// Fallback Scalar Implementations
// ------------------------------------------------------------------------------------
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[inline]
pub fn euclidean_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_equivalence() {
        let v1: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let v2: Vec<f32> = (0..128).map(|i| ((128 - i) as f32) * 0.05).collect();

        let scalar = dot_product_scalar(&v1, &v2);
        let simd = dot_product(&v1, &v2);

        assert!(
            (scalar - simd).abs() < 1e-2,
            "Scalar {scalar} vs SIMD {simd}"
        );
    }

    #[test]
    fn test_euclidean_equivalence() {
        let v1: Vec<f32> = (0..131).map(|i| (i as f32) * 0.07).collect();
        let v2: Vec<f32> = (0..131).map(|i| ((131 - i) as f32) * 0.03).collect();

        let scalar = euclidean_sq_scalar(&v1, &v2);
        let simd = euclidean_distance_sq(&v1, &v2);

        assert!(
            (scalar - simd).abs() < 1e-2,
            "Scalar {scalar} vs SIMD {simd}"
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
        assert!((cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-5);
    }
}
