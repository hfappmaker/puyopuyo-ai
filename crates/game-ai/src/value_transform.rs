/// MuZero-style invertible value transform.
///
/// Compresses large values (e.g., cumulative discounted rewards ranging 0–thousands)
/// into a smaller scale for stable MSE training, without saturation (unlike sigmoid/tanh).
///
/// This is an optional utility — games with bounded rewards (e.g., win/loss: -1/0/+1)
/// may not need value transformation at all.
///
/// - Training: `loss = MSE(network_output, value_transform(target, scale))`
/// - Inference: `value = value_inverse_transform(network_output, scale)`
const EPSILON: f32 = 1e-3;

/// Forward transform: h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x
pub fn value_transform_raw(x: f32) -> f32 {
    x.signum() * ((x.abs() + 1.0).sqrt() - 1.0) + EPSILON * x
}

/// Inverse transform: recovers original value from transformed value.
///
/// Solves y = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x for x.
/// For positive case: y = sqrt(x+1) - 1 + epsilon * x
///   Let u = sqrt(x+1), then x = u^2 - 1, and:
///   y = u - 1 + epsilon * (u^2 - 1)
///   epsilon * u^2 + u - (1 + y + epsilon) = 0
///   u = (-1 + sqrt(1 + 4*epsilon*(1 + y + epsilon))) / (2*epsilon)
///   x = u^2 - 1
pub fn value_inverse_transform_raw(y: f32) -> f32 {
    let sign_y = y.signum();
    let abs_y = y.abs();

    let c = 1.0 + abs_y + EPSILON;
    let discriminant = 1.0 + 4.0 * EPSILON * c;
    let u = (discriminant.sqrt() - 1.0) / (2.0 * EPSILON);
    let x = u * u - 1.0;

    sign_y * x.max(0.0)
}

/// Forward transform with scale normalization.
/// `scale` controls the output range (game-specific tuning parameter).
pub fn value_transform(x: f32, scale: f32) -> f32 {
    value_transform_raw(x) / scale
}

/// Inverse transform with scale denormalization.
/// `scale` must match the value used during training.
pub fn value_inverse_transform(y: f32, scale: f32) -> f32 {
    value_inverse_transform_raw(y * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SCALE: f32 = 15.0;

    #[test]
    fn test_transform_zero() {
        assert!((value_transform(0.0, TEST_SCALE)).abs() < 1e-6);
    }

    #[test]
    fn test_transform_normalized_range() {
        let h100 = value_transform(100.0, TEST_SCALE);
        let h5000 = value_transform(5000.0, TEST_SCALE);
        let h50000 = value_transform(50000.0, TEST_SCALE);
        // After /15 scaling, values should be in roughly 0–20 range
        assert!(h100 < 2.0, "h(100)={} should be < 2.0", h100);
        assert!(h5000 < 6.0, "h(5000)={} should be < 6.0", h5000);
        assert!(h50000 < 20.0, "h(50000)={} should be < 20.0", h50000);
    }

    #[test]
    fn test_roundtrip() {
        let test_values = [0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0, -1.0, -100.0];
        for &x in &test_values {
            let y = value_transform(x, TEST_SCALE);
            let recovered = value_inverse_transform(y, TEST_SCALE);
            assert!(
                (recovered - x).abs() < 0.1,
                "roundtrip failed for x={}: transform={}, inverse={}",
                x,
                y,
                recovered,
            );
        }
    }

    #[test]
    fn test_monotonic() {
        let values = [0.0, 1.0, 10.0, 100.0, 1000.0];
        for w in values.windows(2) {
            assert!(
                value_transform(w[1], TEST_SCALE) > value_transform(w[0], TEST_SCALE),
                "not monotonic: h({}) = {} <= h({}) = {}",
                w[1],
                value_transform(w[1], TEST_SCALE),
                w[0],
                value_transform(w[0], TEST_SCALE),
            );
        }
    }
}
