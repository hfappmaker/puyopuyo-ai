use crate::board::{PuyoColor, NUM_COLORS};
use crate::piece::Piece;

/// splitmix64 finalizer — mixes a u64 seed into a well-distributed hash.
fn splitmix64(s: u64) -> u64 {
    let s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
    s ^ (s >> 31)
}

/// 現在時刻からランダムなシードを生成する。
pub fn time_seed() -> u64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        splitmix64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        splitmix64((js_sys::Date::now() * 1_000_000.0) as u64)
    }
}

/// time_seed() を使ってランダムなピースを生成する。
pub fn random_piece() -> Piece {
    let mut x = time_seed();
    let axis = ((x % NUM_COLORS as u64) as u8) + 1;
    x = (x ^ (x >> 30)).wrapping_mul(0x517cc1b727220a95);
    x = x ^ (x >> 27);
    let sat = ((x % NUM_COLORS as u64) as u8) + 1;
    Piece::new(PuyoColor::from_u8(axis), PuyoColor::from_u8(sat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::PuyoColor;

    #[test]
    fn test_random_piece_valid() {
        for _ in 0..100 {
            let p = random_piece();
            assert_ne!(p.axis_color, PuyoColor::Empty);
            assert_ne!(p.satellite_color, PuyoColor::Empty);
        }
    }

    #[test]
    fn test_time_seed_nonzero() {
        let s = time_seed();
        assert_ne!(s, 0);
    }
}
