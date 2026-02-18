/// Deterministic xorshift64 random number generator.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Generate next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random number in [0, n).
    pub fn next_range(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_range() {
        let mut rng = Rng::new(12345);
        for _ in 0..1000 {
            let v = rng.next_range(4);
            assert!(v < 4);
        }
    }

    #[test]
    fn test_zero_seed_handled() {
        let mut rng = Rng::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }
}
