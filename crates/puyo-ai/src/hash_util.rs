/// splitmix64 finalizer — mixes a u64 seed into a well-distributed hash.
pub fn splitmix64(s: u64) -> u64 {
    let s = (s ^ (s >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let s = (s ^ (s >> 27)).wrapping_mul(0x94D049BB133111EB);
    s ^ (s >> 31)
}

/// 現在時刻からランダムなシードを生成する。
pub fn time_seed() -> u64 {
    splitmix64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
    )
}
