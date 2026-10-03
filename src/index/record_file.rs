//! [`RecordFile`]: one reader for every fixed-size-record index.
//!
//! Every index the toolkit produces is a flat run of equally sized records
//! with (usually) a little-endian `u64` sort key somewhere inside each
//! record.  Instead of nine hand-written readers with their own binary
//! searches, each index defines an [`Entry`] (how to decode one record and
//! where its key is) and reads through `RecordFile<E>`.
//!
//! Writing goes through [`RecordWriter`], which streams entries into a
//! staged [`StoreWriter`] and can sort them in place before committing.

use crate::hprof::HprofError;
use crate::index::store::StoreWriter;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;

// ── Entry ─────────────────────────────────────────────────────────────────────

/// One fixed-size record in an index.
pub trait Entry: Copy + Send + Sync + 'static {
    /// Byte size of one record.
    const SIZE: usize;
    /// Byte offset of the little-endian `u64` sort key inside a record.
    const KEY_OFFSET: usize;

    /// Decode a record.  `bytes.len() == Self::SIZE`.
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Encode a record.  `out.len() == Self::SIZE`; every byte must be
    /// written (padding included).
    fn write_to(&self, out: &mut [u8]);

    /// The sort key.  Must equal the `u64` stored at [`Self::KEY_OFFSET`].
    fn key(&self) -> u64;
}

/// Largest record size any [`Entry`] may declare (scratch buffer bound).
pub const MAX_ENTRY_SIZE: usize = 64;

// ── RecordFile ────────────────────────────────────────────────────────────────

/// Read-only view of a record file, borrowed from a byte source.
///
/// Lookups ([`find`](Self::find), [`range`](Self::range),
/// [`lower_bound`](Self::lower_bound)) require the records to be sorted
/// ascending by key; iteration does not.
pub struct RecordFile<'a, E: Entry> {
    data: &'a [u8],
    _e: PhantomData<E>,
}

impl<E: Entry> Clone for RecordFile<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Entry> Copy for RecordFile<'_, E> {}

impl<'a, E: Entry> RecordFile<'a, E> {
    /// Validate that `data` holds whole records and wrap it.
    pub fn new(data: &'a [u8]) -> Result<Self, HprofError> {
        if !data.len().is_multiple_of(E::SIZE) {
            return Err(HprofError::InvalidIndexFile);
        }
        Ok(Self::from_slice(data))
    }

    /// Wrap data already known to hold whole records (checked in debug builds).
    pub fn from_slice(data: &'a [u8]) -> Self {
        debug_assert!(data.len().is_multiple_of(E::SIZE));
        Self {
            data,
            _e: PhantomData,
        }
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.data.len() / E::SIZE
    }

    /// `true` when there are no records.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Decode record `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<E> {
        let start = i.checked_mul(E::SIZE)?;
        let end = start.checked_add(E::SIZE)?;
        self.data.get(start..end).map(E::from_bytes)
    }

    /// Read only the key of record `i` (no full decode).  `i < len()`.
    pub fn key_at(&self, i: usize) -> u64 {
        read_u64_le(self.data, i * E::SIZE + E::KEY_OFFSET)
    }

    /// Iterate all records in file order.
    pub fn iter(&self) -> RecordIter<'a, E> {
        RecordIter {
            data: self.data,
            pos: 0,
            end: self.len(),
            _e: PhantomData,
        }
    }

    /// Iterate records `start..end` (clamped to the file) in file order.
    pub fn iter_range(&self, start: usize, end: usize) -> RecordIter<'a, E> {
        let end = end.min(self.len());
        RecordIter {
            data: self.data,
            pos: start.min(end),
            end,
            _e: PhantomData,
        }
    }

    /// Iterate all records in parallel (rayon).
    #[cfg(test)]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = E> + use<'a, E> {
        let data = self.data;
        (0..self.len())
            .into_par_iter()
            .map(move |i| E::from_bytes(&data[i * E::SIZE..(i + 1) * E::SIZE]))
    }

    /// Index of the first record whose key is `>= key` (may be `len()`).
    ///
    /// Requires ascending key order.
    pub fn lower_bound(&self, key: u64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_at(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Index one past the last record whose key is `<= key`.
    ///
    /// Requires ascending key order.
    #[cfg(test)]
    pub fn upper_bound(&self, key: u64) -> usize {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.key_at(mid) <= key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The first record with exactly `key`, if any.
    ///
    /// Requires ascending key order.
    pub fn find(&self, key: u64) -> Option<E> {
        let i = self.lower_bound(key);
        if i < self.len() && self.key_at(i) == key {
            self.get(i)
        } else {
            None
        }
    }

    /// All records with exactly `key`, in file order.
    ///
    /// Requires ascending key order.
    pub fn range(&self, key: u64) -> RecordIter<'a, E> {
        let start = self.lower_bound(key);
        let mut end = start;
        let n = self.len();
        while end < n && self.key_at(end) == key {
            end += 1;
        }
        RecordIter {
            data: self.data,
            pos: start,
            end,
            _e: PhantomData,
        }
    }

    /// All records with exactly `key`, in parallel (rayon).
    ///
    /// Requires ascending key order.
    pub fn par_range(&self, key: u64) -> impl ParallelIterator<Item = E> + use<'a, E> {
        let range = self.range(key);
        let data = self.data;
        (range.pos..range.end)
            .into_par_iter()
            .map(move |i| E::from_bytes(&data[i * E::SIZE..(i + 1) * E::SIZE]))
    }

    /// All records with `lo <= key <= hi`, in file order.
    ///
    /// Requires ascending key order.
    #[cfg(test)]
    pub fn range_between(&self, lo: u64, hi: u64) -> RecordIter<'a, E> {
        let start = self.lower_bound(lo);
        let end = self.upper_bound(hi).max(start);
        RecordIter {
            data: self.data,
            pos: start,
            end,
            _e: PhantomData,
        }
    }
}

// ── RecordIter ────────────────────────────────────────────────────────────────

/// Iterator over decoded fixed-size index records, in file order.
pub struct RecordIter<'a, E: Entry> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    _e: PhantomData<E>,
}

impl<E: Entry> Iterator for RecordIter<'_, E> {
    type Item = E;

    fn next(&mut self) -> Option<E> {
        if self.pos >= self.end {
            return None;
        }
        let start = self.pos * E::SIZE;
        let e = E::from_bytes(&self.data[start..start + E::SIZE]);
        self.pos += 1;
        Some(e)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.end - self.pos;
        (n, Some(n))
    }

    /// Jump straight to record `n` instead of decoding the ones skipped, so
    /// `skip(offset)` in a paged query is O(1).
    fn nth(&mut self, n: usize) -> Option<E> {
        self.pos = self.pos.saturating_add(n).min(self.end);
        self.next()
    }
}

impl<E: Entry> ExactSizeIterator for RecordIter<'_, E> {}

impl<E: Entry> DoubleEndedIterator for RecordIter<'_, E> {
    fn next_back(&mut self) -> Option<E> {
        if self.pos >= self.end {
            return None;
        }
        self.end -= 1;
        let start = self.end * E::SIZE;
        Some(E::from_bytes(&self.data[start..start + E::SIZE]))
    }

    fn nth_back(&mut self, n: usize) -> Option<E> {
        self.end = self.end.saturating_sub(n).max(self.pos);
        self.next_back()
    }
}

// ── RecordWriter ──────────────────────────────────────────────────────────────

/// Streams records into a staged store entry.
///
/// Call [`finish`](Self::finish) when the records were pushed in key order
/// (or order does not matter), [`finish_sorted`](Self::finish_sorted) to sort
/// ascending by key in place before committing, or
/// [`finish_sorted_desc`](Self::finish_sorted_desc) for descending order.
pub struct RecordWriter<E: Entry> {
    inner: Box<dyn StoreWriter>,
    count: u64,
    _e: PhantomData<E>,
}

impl<E: Entry> RecordWriter<E> {
    pub fn new(inner: Box<dyn StoreWriter>) -> Self {
        debug_assert!(E::SIZE <= MAX_ENTRY_SIZE);
        Self {
            inner,
            count: 0,
            _e: PhantomData,
        }
    }

    /// Append one record.
    pub fn push(&mut self, e: &E) -> Result<(), HprofError> {
        let mut buf = [0u8; MAX_ENTRY_SIZE];
        e.write_to(&mut buf[..E::SIZE]);
        self.inner.write_all(&buf[..E::SIZE])?;
        self.count += 1;
        Ok(())
    }

    /// Records pushed so far.
    #[cfg(test)]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Commit as written.  Returns the record count.
    pub fn finish(mut self) -> Result<u64, HprofError> {
        self.inner.flush()?;
        self.inner.commit()?;
        Ok(self.count)
    }

    /// Sort ascending by key in place, then commit.  Returns the record count.
    pub fn finish_sorted(self) -> Result<u64, HprofError> {
        self.finish_with_order(None, false)
    }

    /// Sort descending by key in place, then commit.  Returns the record count.
    pub fn finish_sorted_desc(self) -> Result<u64, HprofError> {
        self.finish_with_order(None, true)
    }

    /// Sort ascending by `(key, secondary)` where `secondary` is the byte
    /// offset of a second little-endian `u64` inside the record, then commit.
    pub fn finish_sorted_then_by(self, secondary_key_offset: usize) -> Result<u64, HprofError> {
        self.finish_with_order(Some(secondary_key_offset), false)
    }

    /// Sort by `(key, secondary)` in the given direction, then commit.
    ///
    /// Use a secondary key whenever ties are possible and the output must be
    /// byte-deterministic (the sort is unstable).
    pub fn finish_sorted_by(
        self,
        secondary_key_offset: Option<usize>,
        descending: bool,
    ) -> Result<u64, HprofError> {
        self.finish_with_order(secondary_key_offset, descending)
    }

    fn finish_with_order(
        mut self,
        secondary_key_offset: Option<usize>,
        descending: bool,
    ) -> Result<u64, HprofError> {
        if self.count > 1 {
            let bytes = self.inner.as_mut_bytes()?;
            crate::sort::parallel_introsort_by_keys(
                bytes,
                E::SIZE,
                E::KEY_OFFSET,
                secondary_key_offset,
                descending,
            );
        }
        self.inner.commit()?;
        Ok(self.count)
    }
}

// ── Byte helpers ──────────────────────────────────────────────────────────────

/// Read a little-endian `u64` at `off`.  `off + 8 <= data.len()`.
#[inline]
pub fn read_u64_le(data: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&data[off..off + 8]);
    u64::from_le_bytes(b)
}

/// Read a little-endian `u32` at `off`.  `off + 4 <= data.len()`.
#[inline]
pub fn read_u32_le(data: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&data[off..off + 4]);
    u32::from_le_bytes(b)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexStore, MemStore};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Pair {
        key: u64,
        value: u64,
    }

    impl Entry for Pair {
        const SIZE: usize = 16;
        const KEY_OFFSET: usize = 0;
        fn from_bytes(b: &[u8]) -> Self {
            Self {
                key: read_u64_le(b, 0),
                value: read_u64_le(b, 8),
            }
        }
        fn write_to(&self, out: &mut [u8]) {
            out[0..8].copy_from_slice(&self.key.to_le_bytes());
            out[8..16].copy_from_slice(&self.value.to_le_bytes());
        }
        fn key(&self) -> u64 {
            self.key
        }
    }

    fn file_of(pairs: &[(u64, u64)]) -> Vec<u8> {
        let mut v = vec![0u8; pairs.len() * 16];
        for (i, &(k, val)) in pairs.iter().enumerate() {
            Pair { key: k, value: val }.write_to(&mut v[i * 16..(i + 1) * 16]);
        }
        v
    }

    #[test]
    fn validates_size() {
        assert!(RecordFile::<Pair>::new(&[0u8; 15]).is_err());
        assert!(RecordFile::<Pair>::new(&[0u8; 32]).is_ok());
        assert!(RecordFile::<Pair>::new(&[]).unwrap().is_empty());
    }

    #[test]
    fn get_iter_and_bounds() {
        let data = file_of(&[(1, 10), (2, 20), (2, 21), (5, 50)]);
        let f = RecordFile::<Pair>::new(&data).unwrap();
        assert_eq!(f.len(), 4);
        assert_eq!(f.get(3), Some(Pair { key: 5, value: 50 }));
        assert_eq!(f.get(4), None);
        assert_eq!(
            f.iter().map(|p| p.value).collect::<Vec<_>>(),
            vec![10, 20, 21, 50]
        );
        assert_eq!(f.iter().next_back().map(|p| p.key), Some(5));
        assert_eq!(f.iter().len(), 4);
        assert_eq!(f.lower_bound(0), 0);
        assert_eq!(f.lower_bound(2), 1);
        assert_eq!(f.lower_bound(3), 3);
        assert_eq!(f.lower_bound(6), 4);
        assert_eq!(f.upper_bound(2), 3);
    }

    #[test]
    fn find_and_range() {
        let data = file_of(&[(1, 10), (2, 20), (2, 21), (5, 50)]);
        let f = RecordFile::<Pair>::new(&data).unwrap();
        assert_eq!(f.find(2).map(|p| p.value), Some(20));
        assert_eq!(f.find(3), None);
        assert_eq!(
            f.range(2).map(|p| p.value).collect::<Vec<_>>(),
            vec![20, 21]
        );
        assert!(f.range(3).next().is_none());
        assert_eq!(
            f.range_between(2, 5).map(|p| p.value).collect::<Vec<_>>(),
            vec![20, 21, 50]
        );
        assert!(f.range_between(3, 4).next().is_none());
        let par: u64 = f.par_iter().map(|p| p.value).sum();
        assert_eq!(par, 101);
    }

    #[test]
    fn skipping_jumps_without_decoding() {
        let data = file_of(&[(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
        let f = RecordFile::<Pair>::new(&data).unwrap();
        for n in 0..8 {
            assert_eq!(f.iter().nth(n), f.get(n), "nth({n})");
            assert_eq!(
                f.iter().skip(n).map(|p| p.key).collect::<Vec<_>>(),
                (1..=5u64).skip(n).collect::<Vec<_>>(),
                "skip({n}) then collect"
            );
        }
        let mut it = f.iter();
        assert_eq!(it.nth(1).map(|p| p.key), Some(2));
        assert_eq!(it.next().map(|p| p.key), Some(3));
        assert_eq!(it.len(), 2);
        let mut back = f.iter();
        assert_eq!(back.nth_back(1).map(|p| p.key), Some(4));
        assert_eq!(back.len(), 3);
        assert_eq!(back.nth_back(10), None);
        assert_eq!(f.iter().nth(99), None);
    }

    #[test]
    fn writer_sorts_and_commits() {
        let store = MemStore::new();
        let mut w = RecordWriter::<Pair>::new(store.create("p.bin").unwrap());
        for (k, v) in [(5u64, 1u64), (1, 2), (3, 3)] {
            w.push(&Pair { key: k, value: v }).unwrap();
        }
        assert_eq!(w.count(), 3);
        assert_eq!(w.finish_sorted().unwrap(), 3);
        let src = store.open("p.bin").unwrap();
        let f = RecordFile::<Pair>::new(src.as_ref()).unwrap();
        assert_eq!(f.iter().map(|p| p.key).collect::<Vec<_>>(), vec![1, 3, 5]);
        assert_eq!(f.find(3).unwrap().value, 3);

        let mut w = RecordWriter::<Pair>::new(store.create("d.bin").unwrap());
        for k in [5u64, 1, 3] {
            w.push(&Pair { key: k, value: 0 }).unwrap();
        }
        w.finish_sorted_desc().unwrap();
        let src = store.open("d.bin").unwrap();
        let f = RecordFile::<Pair>::new(src.as_ref()).unwrap();
        assert_eq!(f.iter().map(|p| p.key).collect::<Vec<_>>(), vec![5, 3, 1]);

        let w = RecordWriter::<Pair>::new(store.create("e.bin").unwrap());
        assert_eq!(w.finish().unwrap(), 0);
        assert!(store.open("e.bin").unwrap().is_empty());
    }
}
