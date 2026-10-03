//! Generic parallel in-place introsort for fixed-size binary records.
//!
//! Records are sorted by a little-endian `u64` key at a configurable byte
//! offset within each record — ascending by default, descending via
//! [`parallel_introsort_by_keys`] (which also takes an optional secondary
//! `u64` key).  Uses `rayon::join` for parallelism; falls
//! back to heapsort for small partitions.  This is the only sort in the crate:
//! every index builder sorts through it, on an `MmapMut` or a `Vec<u8>` alike.

use crate::index::read_u64_le;

/// Sort `data` in place, ascending by key.
///
/// * `record_size` — byte length of each record (must evenly divide `data.len()`)
/// * `key_offset`  — byte offset of the 8-byte little-endian sort key within
///   each record (must satisfy `key_offset + 8 <= record_size`)
pub fn parallel_introsort(data: &mut [u8], record_size: usize, key_offset: usize) {
    parallel_introsort_by_keys(data, record_size, key_offset, None, false);
}

/// Sort `data` in place by `(key, secondary key)`.
///
/// Both keys are little-endian `u64`s inside the record; the secondary key
/// (if any) breaks ties.  `descending` applies to both.
pub fn parallel_introsort_by_keys(
    data: &mut [u8],
    record_size: usize,
    key_offset: usize,
    secondary_key_offset: Option<usize>,
    descending: bool,
) {
    let n = data.len() / record_size;
    if n < 2 {
        return;
    }
    let layout = Layout {
        record_size,
        key_offset,
        secondary_key_offset,
        descending,
    };
    let depth_limit = 2 * (usize::BITS - n.leading_zeros()) as usize;
    introsort(data, n, layout, depth_limit);
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Below this many records, use serial heapsort.
const PARALLEL_THRESHOLD: usize = 4096;

// ── Layout ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Layout {
    record_size: usize,
    key_offset: usize,
    secondary_key_offset: Option<usize>,
    descending: bool,
}

impl Layout {
    /// The comparison key of record `idx` (both parts bit-inverted when
    /// descending, so that an ascending sort on it yields descending order).
    #[inline]
    fn key(&self, data: &[u8], idx: usize) -> (u64, u64) {
        let base = idx * self.record_size;
        let k1 = read_u64_le(data, base + self.key_offset);
        let k2 = self
            .secondary_key_offset
            .map(|off| read_u64_le(data, base + off))
            .unwrap_or(0);
        if self.descending {
            (!k1, !k2)
        } else {
            (k1, k2)
        }
    }

    #[inline]
    fn swap(&self, data: &mut [u8], i: usize, j: usize) {
        if i == j {
            return;
        }
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let a_start = a * self.record_size;
        let b_start = b * self.record_size;
        let (left, right) = data.split_at_mut(b_start);
        left[a_start..a_start + self.record_size].swap_with_slice(&mut right[..self.record_size]);
    }
}

// ── Introsort ─────────────────────────────────────────────────────────────────

fn introsort(data: &mut [u8], n: usize, l: Layout, depth_limit: usize) {
    if n <= 1 {
        return;
    }
    if n <= PARALLEL_THRESHOLD || depth_limit == 0 {
        heapsort(data, n, l);
        return;
    }

    let pivot = partition(data, n, l);
    let (left, rest) = data.split_at_mut(pivot * l.record_size);
    let right = &mut rest[l.record_size..];
    let right_n = n - pivot - 1;
    let next = depth_limit - 1;

    rayon::join(
        || introsort(left, pivot, l, next),
        || introsort(right, right_n, l, next),
    );
}

// ── Heapsort ──────────────────────────────────────────────────────────────────

fn heapsort(data: &mut [u8], n: usize, l: Layout) {
    if n < 2 {
        return;
    }
    let mut i = n / 2;
    while i > 0 {
        i -= 1;
        sift_down(data, i, n, l);
    }
    let mut end = n;
    while end > 1 {
        end -= 1;
        l.swap(data, 0, end);
        sift_down(data, 0, end, l);
    }
}

fn sift_down(data: &mut [u8], mut root: usize, n: usize, l: Layout) {
    loop {
        let left = 2 * root + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let largest = if right < n && l.key(data, right) > l.key(data, left) {
            right
        } else {
            left
        };
        if l.key(data, largest) <= l.key(data, root) {
            break;
        }
        l.swap(data, root, largest);
        root = largest;
    }
}

// ── Lomuto partition with median-of-three pivot ───────────────────────────────

fn partition(data: &mut [u8], n: usize, l: Layout) -> usize {
    let last = n - 1;
    let mid = n / 2;
    if l.key(data, mid) < l.key(data, 0) {
        l.swap(data, 0, mid);
    }
    if l.key(data, last) < l.key(data, 0) {
        l.swap(data, 0, last);
    }
    if l.key(data, mid) > l.key(data, last) {
        l.swap(data, mid, last);
    }
    let pivot_val = l.key(data, last);
    let mut store = 0usize;
    for i in 0..last {
        if l.key(data, i) <= pivot_val {
            l.swap(data, i, store);
            store += 1;
        }
    }
    l.swap(data, store, last);
    store
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-byte records: key at `key_offset`, payload = key * 3 in the other slot.
    fn records(keys: &[u64], key_offset: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(keys.len() * 16);
        for &k in keys {
            let mut rec = [0u8; 16];
            let other = if key_offset == 0 { 8 } else { 0 };
            rec[key_offset..key_offset + 8].copy_from_slice(&k.to_le_bytes());
            rec[other..other + 8].copy_from_slice(&k.wrapping_mul(3).to_le_bytes());
            v.extend_from_slice(&rec);
        }
        v
    }

    fn keys_of(data: &[u8], key_offset: usize) -> Vec<u64> {
        data.as_chunks::<16>()
            .0
            .iter()
            .map(|r| read_u64_le(r, key_offset))
            .collect()
    }

    fn payloads_consistent(data: &[u8], key_offset: usize) -> bool {
        let other = if key_offset == 0 { 8 } else { 0 };
        data.as_chunks::<16>()
            .0
            .iter()
            .all(|r| read_u64_le(r, other) == read_u64_le(r, key_offset).wrapping_mul(3))
    }

    #[test]
    fn sorts_small_ascending_and_keeps_payload() {
        let mut data = records(&[5, 3, 8, 1, 9, 2, 7, 4, 6], 8);
        parallel_introsort(&mut data, 16, 8);
        assert_eq!(keys_of(&data, 8), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(payloads_consistent(&data, 8));
    }

    #[test]
    fn sorts_descending() {
        let mut data = records(&[5, 3, 8, 1, 0, u64::MAX], 0);
        parallel_introsort_by_keys(&mut data, 16, 0, None, true);
        assert_eq!(keys_of(&data, 0), vec![u64::MAX, 8, 5, 3, 1, 0]);
        assert!(payloads_consistent(&data, 0));
    }

    #[test]
    fn sorts_by_two_keys() {
        // (k1, k2) pairs in 16-byte records.
        let pairs: [(u64, u64); 6] = [(2, 9), (1, 5), (2, 1), (1, 7), (0, 3), (2, 4)];
        let mut data = Vec::new();
        for (a, b) in pairs {
            data.extend_from_slice(&a.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
        }
        parallel_introsort_by_keys(&mut data, 16, 0, Some(8), false);
        let sorted: Vec<(u64, u64)> = data
            .as_chunks::<16>()
            .0
            .iter()
            .map(|r| (read_u64_le(r, 0), read_u64_le(r, 8)))
            .collect();
        assert_eq!(sorted, vec![(0, 3), (1, 5), (1, 7), (2, 1), (2, 4), (2, 9)]);

        parallel_introsort_by_keys(&mut data, 16, 0, Some(8), true);
        let sorted: Vec<(u64, u64)> = data
            .as_chunks::<16>()
            .0
            .iter()
            .map(|r| (read_u64_le(r, 0), read_u64_le(r, 8)))
            .collect();
        assert_eq!(sorted, vec![(2, 9), (2, 4), (2, 1), (1, 7), (1, 5), (0, 3)]);
    }

    #[test]
    fn sorts_large_input_through_parallel_path() {
        // Deterministic pseudo-random keys, more than PARALLEL_THRESHOLD.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let keys: Vec<u64> = (0..20_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % 5000 // plenty of duplicates
            })
            .collect();
        let mut data = records(&keys, 0);
        parallel_introsort(&mut data, 16, 0);
        let sorted = keys_of(&data, 0);
        assert!(sorted.windows(2).all(|w| w[0] <= w[1]));
        let mut expected = keys.clone();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        assert!(payloads_consistent(&data, 0));
    }

    #[test]
    fn tiny_inputs_are_no_ops() {
        let mut empty: Vec<u8> = Vec::new();
        parallel_introsort(&mut empty, 16, 0);
        let mut one = records(&[42], 0);
        parallel_introsort(&mut one, 16, 0);
        assert_eq!(keys_of(&one, 0), vec![42]);
    }
}
