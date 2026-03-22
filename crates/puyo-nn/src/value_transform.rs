/// MuZero-style invertible value transform.
///
/// Compresses large values (e.g., cumulative discounted rewards ranging 0–thousands)
/// into a smaller scale for stable MSE training, without saturation (unlike sigmoid/tanh).
///
/// - Training: `loss = MSE(network_output, value_transform(target))`
/// - Inference: `value = value_inverse_transform(network_output)`
const EPSILON: f32 = 1e-3;

/// Scale factor to normalize transformed values to approximately 0–1 range.
/// Cumulative discounted rewards can reach ~50,000, whose transform is ~273.
/// Dividing by 15 keeps typical values in a manageable range for stable training.
pub const VALUE_SCALE: f32 = 15.0;

/// Forward transform: h(x) = sign(x) * (sqrt(|x| + 1) - 1) + epsilon * x
fn value_transform_raw(x: f32) -> f32 {
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
fn value_inverse_transform_raw(y: f32) -> f32 {
    let sign_y = y.signum();
    let abs_y = y.abs();

    let c = 1.0 + abs_y + EPSILON;
    let discriminant = 1.0 + 4.0 * EPSILON * c;
    let u = (discriminant.sqrt() - 1.0) / (2.0 * EPSILON);
    let x = u * u - 1.0;

    sign_y * x.max(0.0)
}

/// Forward transform with scale normalization.
/// Maps cumulative rewards to approximately 0–1 range for stable training.
pub fn value_transform(x: f32) -> f32 {
    value_transform_raw(x) / VALUE_SCALE
}

/// Inverse transform with scale denormalization.
/// Recovers original value from scaled network output.
pub fn value_inverse_transform(y: f32) -> f32 {
    value_inverse_transform_raw(y * VALUE_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_zero() {
        assert!((value_transform(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_transform_normalized_range() {
        let h100 = value_transform(100.0);
        let h5000 = value_transform(5000.0);
        let h50000 = value_transform(50000.0);
        // After /15 scaling, values should be in roughly 0–20 range
        assert!(h100 < 2.0, "h(100)={} should be < 2.0", h100);
        assert!(h5000 < 6.0, "h(5000)={} should be < 6.0", h5000);
        assert!(h50000 < 20.0, "h(50000)={} should be < 20.0", h50000);
    }

    #[test]
    fn test_roundtrip() {
        let test_values = [0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0, -1.0, -100.0];
        for &x in &test_values {
            let y = value_transform(x);
            let recovered = value_inverse_transform(y);
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
                value_transform(w[1]) > value_transform(w[0]),
                "not monotonic: h({}) = {} <= h({}) = {}",
                w[1],
                value_transform(w[1]),
                w[0],
                value_transform(w[0]),
            );
        }
    }
}
