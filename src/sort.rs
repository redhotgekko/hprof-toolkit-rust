//! Generic parallel in-place introsort for fixed-size binary records.
//!
//! Records are sorted ascending by a little-endian `u64` key at a
//! configurable byte offset within each record.  Uses `rayon::join` for
//! parallelism; falls back to heapsort for small partitions.

/// Sort `data` in-place, treating it as an array of fixed-size records.
///
/// * `record_size` — byte length of each record (must evenly divide `data.len()`)
/// * `key_offset`  — byte offset of the 8-byte little-endian sort key within
///   each record (must satisfy `key_offset + 8 <= record_size`)
pub fn parallel_introsort(data: &mut [u8], record_size: usize, key_offset: usize) {
    let n = data.len() / record_size;
    if n < 2 {
        return;
    }
    let depth_limit = 2 * (usize::BITS - n.leading_zeros()) as usize;
    introsort(data, n, record_size, key_offset, depth_limit);
}

/// Exposed for unit-test shims in sibling modules that test a fixed
/// record-size/key-offset heapsort under its old name.
pub fn heapsort_for_tests(data: &mut [u8], n: usize, record_size: usize, key_offset: usize) {
    heapsort(data, n, record_size, key_offset);
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Below this many records, use serial heapsort.
const PARALLEL_THRESHOLD: usize = 4096;

// ── Introsort ─────────────────────────────────────────────────────────────────

fn introsort(data: &mut [u8], n: usize, record_size: usize, key_offset: usize, depth_limit: usize) {
    if n <= 1 {
        return;
    }
    if n <= PARALLEL_THRESHOLD || depth_limit == 0 {
        heapsort(data, n, record_size, key_offset);
        return;
    }

    let pivot = partition(data, n, record_size, key_offset);
    let (left, rest) = data.split_at_mut(pivot * record_size);
    let right = &mut rest[record_size..];
    let right_n = n - pivot - 1;
    let next = depth_limit - 1;

    rayon::join(
        || introsort(left, pivot, record_size, key_offset, next),
        || introsort(right, right_n, record_size, key_offset, next),
    );
}

// ── Heapsort ──────────────────────────────────────────────────────────────────

fn heapsort(data: &mut [u8], n: usize, record_size: usize, key_offset: usize) {
    if n < 2 {
        return;
    }
    let mut i = n / 2;
    while i > 0 {
        i -= 1;
        sift_down(data, i, n, record_size, key_offset);
    }
    let mut end = n;
    while end > 1 {
        end -= 1;
        swap_records(data, 0, end, record_size);
        sift_down(data, 0, end, record_size, key_offset);
    }
}

fn sift_down(data: &mut [u8], mut root: usize, n: usize, record_size: usize, key_offset: usize) {
    loop {
        let left = 2 * root + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let largest = if right < n
            && read_key(data, right, record_size, key_offset)
                > read_key(data, left, record_size, key_offset)
        {
            right
        } else {
            left
        };
        if read_key(data, largest, record_size, key_offset)
            <= read_key(data, root, record_size, key_offset)
        {
            break;
        }
        swap_records(data, root, largest, record_size);
        root = largest;
    }
}

// ── Lomuto partition with median-of-three pivot ───────────────────────────────

fn partition(data: &mut [u8], n: usize, record_size: usize, key_offset: usize) -> usize {
    let last = n - 1;
    let mid = n / 2;
    if read_key(data, mid, record_size, key_offset) < read_key(data, 0, record_size, key_offset) {
        swap_records(data, 0, mid, record_size);
    }
    if read_key(data, last, record_size, key_offset) < read_key(data, 0, record_size, key_offset) {
        swap_records(data, 0, last, record_size);
    }
    if read_key(data, mid, record_size, key_offset) > read_key(data, last, record_size, key_offset)
    {
        swap_records(data, mid, last, record_size);
    }
    let pivot_val = read_key(data, last, record_size, key_offset);
    let mut store = 0usize;
    for i in 0..last {
        if read_key(data, i, record_size, key_offset) <= pivot_val {
            swap_records(data, i, store, record_size);
            store += 1;
        }
    }
    swap_records(data, store, last, record_size);
    store
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn read_key(data: &[u8], idx: usize, record_size: usize, key_offset: usize) -> u64 {
    let start = idx * record_size + key_offset;
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[start..start + 8]);
    u64::from_le_bytes(bytes)
}

fn swap_records(data: &mut [u8], i: usize, j: usize, record_size: usize) {
    if i == j {
        return;
    }
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let a_start = a * record_size;
    let b_start = b * record_size;
    let (left, right) = data.split_at_mut(b_start);
    left[a_start..a_start + record_size].swap_with_slice(&mut right[..record_size]);
}
