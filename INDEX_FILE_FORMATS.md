# Index File Formats

All index files are written to `{hprof_stem}.indexes/` alongside the source `.hprof` file (or to the directory given by `--index-dir` / `IndexOptions::index_dir`).
They are built in pipeline phases and memory-mapped at query time. The same builders write to an in-memory store in tests; the file names below are the entry names in either backend.

**Common properties across all files:**
- No per-file header, magic bytes, or version field — raw records start at offset 0. The layout version and the identity of the hprof they belong to live in the directory's manifest (`index.json`, below).
- All multi-byte integers are **little-endian**
- All files contain fixed-size records; entry count = `file_size / entry_size`
- Files are validated on open: `file_size % entry_size == 0`

---

## index.json (manifest)

Written by the build pipeline after every completed step and read by every consumer. A file that is present in the directory but **not** listed under `built` is treated as absent and rebuilt; a manifest whose `format_version` or hprof identity does not match causes a full rebuild (indexes are derived data, there is no migration).

```json
{
  "format_version": 2,
  "hprof_len": 766710804,
  "id_size": 8,
  "hprof_timestamp_ms": 1758215000000,
  "built": {
    "object_store.bin": 1758216000,
    "refs.bin": 1758216012
  }
}
```

| Field | Meaning |
|---|---|
| `format_version` | Layout version of every file in the directory; `FORMAT_VERSION` in `src/index/manifest.rs`. Bumped whenever any file's layout or name changes. |
| `hprof_len` | Byte length of the hprof the indexes were built from. |
| `id_size` | Identifier size of that hprof (4 or 8). |
| `hprof_timestamp_ms` | Creation timestamp from that hprof's header (ms since the epoch). Together with the length and id size it identifies the dump: two dumps of the same length taken at different times do not share indexes. |
| `built` | File name → Unix timestamp (seconds) of the build step that committed it. |

Source: `src/index/manifest.rs`

**Write protocol.** Every file is written to `<name>.tmp` and renamed into place when its build step succeeds, then the manifest is rewritten. A crash therefore never leaves a truncated file under a final name, and the next run resumes at the first step missing from `built`. Temporary `*.part.NNNN` files (used by parallel builders) are removed when their final file is committed and swept at the start of the step that owns them.

---

## record_index.bin

Phase 1 output. One entry per top-level hprof record (UTF8, LOAD_CLASS, HEAP_DUMP_SEGMENT, etc.).

**Entry size:** 16 bytes

```
offset  size  type      field
0       1     u8        tag           hprof record tag byte
1       3     [u8; 3]   padding       zeros
4       4     u32       body_length   byte length of the record body
8       8     u64       position      byte offset of the tag byte in the hprof file
```

Not sorted. Entries appear in hprof record order.

Source: `src/record_index/entry.rs`

---

## heap_index/ (per-segment files — build intermediate)

Phase 2 output. One file per `HEAP_DUMP` / `HEAP_DUMP_SEGMENT` record, named `heap_index/<hh>/HPROF_HEAP_DUMP[_SEGMENT]_<pos>` where `<pos>` is the record's hex file offset and `<hh>` its first two hex digits. Each file holds one entry per heap sub-record within that segment.

These files are **intermediates**: they are streamed into `object_store.bin` and deleted as soon as it has been committed. A finished index directory does not contain them.

**Entry size:** 24 bytes

```
offset  size  type      field
0       1     u8        tag         heap sub-record tag (see table below)
1       7     [u8; 7]   padding     zeros
8       8     u64       object_id   first id-sized field, zero-extended to 8 bytes
16      8     u64       position    byte offset of the subtag byte in the hprof file
```

Sub-record tag values:

| Tag   | Constant                  | Sub-record type             |
|-------|---------------------------|-----------------------------|
| `0xFF`| `TAG_ROOT_UNKNOWN`        | `GC_ROOT_UNKNOWN`           |
| `0x01`| `TAG_ROOT_JNI_GLOBAL`     | `GC_ROOT_JNI_GLOBAL`        |
| `0x02`| `TAG_ROOT_JNI_LOCAL`      | `GC_ROOT_JNI_LOCAL`         |
| `0x03`| `TAG_ROOT_JAVA_FRAME`     | `GC_ROOT_JAVA_FRAME`        |
| `0x04`| `TAG_ROOT_NATIVE_STACK`   | `GC_ROOT_NATIVE_STACK`      |
| `0x05`| `TAG_ROOT_STICKY_CLASS`   | `GC_ROOT_STICKY_CLASS`      |
| `0x06`| `TAG_ROOT_THREAD_BLOCK`   | `GC_ROOT_THREAD_BLOCK`      |
| `0x07`| `TAG_ROOT_MONITOR_USED`   | `GC_ROOT_MONITOR_USED`      |
| `0x08`| `TAG_ROOT_THREAD_OBJ`     | `GC_ROOT_THREAD_OBJ`        |
| `0x20`| `TAG_CLASS_DUMP`          | `HPROF_GC_CLASS_DUMP`       |
| `0x21`| `TAG_INSTANCE_DUMP`       | `HPROF_GC_INSTANCE_DUMP`    |
| `0x22`| `TAG_OBJ_ARRAY_DUMP`      | `HPROF_GC_OBJ_ARRAY_DUMP`   |
| `0x23`| `TAG_PRIM_ARRAY_DUMP`     | `HPROF_GC_PRIM_ARRAY_DUMP`  |

Not sorted. Entries appear in sub-record order within the segment.

Source: `src/heap_index/sub_record.rs`

---

## object_store.bin

Phase 4 output. All heap sub-record entries from every segment merged into one file.

**Entry size:** 24 bytes — identical layout to the per-segment heap index entries.

```
offset  size  type      field
0       1     u8        tag         heap sub-record tag (same values as above)
1       7     [u8; 7]   padding     zeros
8       8     u64       object_id   sort key
16      8     u64       position    byte offset of the subtag byte in the hprof file
```

Sorted ascending by `object_id`. Enables O(log n) binary search by object ID.

Source: `src/object_store.rs`

---

## utf8.bin

One entry per `HPROF_UTF8` record, mapping name IDs to string data locations in the hprof file.

**Entry size:** 24 bytes

```
offset  size  type      field
0       8     u64       name_id        sort key
8       8     u64       string_start   byte offset of the string data in the hprof file
16      4     u32       string_length  byte length of the string data
20      4     [u8; 4]   padding        zeros
```

Sorted ascending by `name_id`.

Source: `src/heap_query/name_index.rs`

---

## loadclass.bin

One entry per `HPROF_LOAD_CLASS` record, mapping class object IDs to their UTF-8 name IDs.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   class_id       sort key (class object ID)
8       8     u64   class_name_id  name_id in utf8.bin for this class's name
```

Sorted ascending by `class_id`.

Source: `src/heap_query/name_index.rs`

---

## refs.bin

One entry per non-null object reference field found in `INSTANCE_DUMP`, `CLASS_DUMP` (static fields), and `OBJ_ARRAY_DUMP` records.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   to_object_id    sort key (the object being pointed to)
8       8     u64   from_object_id  the object holding the reference
```

Sorted ascending by `to_object_id`. Enables O(log n) reverse-reference lookup.

Source: `src/ref_index.rs`

---

## Root index files

One file per GC root type. Each entry records a GC root by object ID and hprof location.

**Files:**

| File                    | GC root type               | Sub-record tag |
|-------------------------|----------------------------|----------------|
| `root_unknown.bin`      | `GC_ROOT_UNKNOWN`          | `0xFF`         |
| `root_jni_global.bin`   | `GC_ROOT_JNI_GLOBAL`       | `0x01`         |
| `root_jni_local.bin`    | `GC_ROOT_JNI_LOCAL`        | `0x02`         |
| `root_java_frame.bin`   | `GC_ROOT_JAVA_FRAME`       | `0x03`         |
| `root_native_stack.bin` | `GC_ROOT_NATIVE_STACK`     | `0x04`         |
| `root_sticky_class.bin` | `GC_ROOT_STICKY_CLASS`     | `0x05`         |
| `root_thread_block.bin` | `GC_ROOT_THREAD_BLOCK`     | `0x06`         |
| `root_monitor_used.bin` | `GC_ROOT_MONITOR_USED`     | `0x07`         |
| `root_thread_obj.bin`   | `GC_ROOT_THREAD_OBJ`       | `0x08`         |

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   object_id   sort key
8       8     u64   position    byte offset of the subtag byte in the hprof file
```

Each file is sorted ascending by `object_id`.

Source: `src/root_index.rs`

---

## Auxiliary index files

One file per auxiliary top-level record type. All five share the same 16-byte format.

**Files:**

| File                 | Record type           | hprof tag | Key field                      |
|----------------------|-----------------------|-----------|--------------------------------|
| `frames.bin`         | `HPROF_FRAME`         | `0x04`    | `frame_id` (ID, 4 or 8 bytes) |
| `traces.bin`         | `HPROF_TRACE`         | `0x05`    | `trace_serial` (u32)          |
| `start_threads.bin`  | `HPROF_START_THREAD`  | `0x0A`    | `thread_serial` (u32)         |
| `end_threads.bin`    | `HPROF_END_THREAD`    | `0x0B`    | `thread_serial` (u32)         |
| `unload_classes.bin` | `HPROF_UNLOAD_CLASS`  | `0x03`    | `class_serial` (u32)          |

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   key           primary identifier or serial number, zero-extended to u64
8       8     u64   hprof_offset  byte offset of the record tag byte in the hprof file
```

Each file is sorted ascending by `key`.

Source: `src/aux_index.rs`

---

## Per-class files

Both are built in one pass over `object_store.bin`. Objects are grouped by a **class key**, one `u64` per kind of object (`src/class_key.rs`):

| Kind | Encoding |
|---|---|
| instance of class object `id` | `id` |
| object array whose **array class** is `arr` (hprof records the array class, e.g. `[Ljava.lang.String;`, in an object-array record, not the element class) | bit 63 set, `arr` in bits 0–62 |
| primitive array of hprof type `t` | bits 63 and 62 set, `t` in bits 0–7 |

64-bit HotSpot addresses never set the top two bits, so synthetic keys cannot collide with real class ids. GC-root records and class dumps are not grouped (a class dump is not an instance of anything).

### instances_by_class.bin

One entry per instance, object array and primitive array.

**Entry size:** 24 bytes

```
offset  size  type  field
0       8     u64   class_key   primary sort key
8       8     u64   object_id   secondary sort key
16      8     u64   position    byte offset of the sub-record in the hprof file
```

Sorted ascending by `(class_key, object_id)`, so "all instances of X" is one binary search plus a contiguous range, and the count is the range length.

### class_histogram.bin

One entry per class key that has at least one object.

**Entry size:** 24 bytes

```
offset  size  type  field
0       8     u64   instance_count  sort key
8       8     u64   shallow_bytes   instance field bytes / array element bytes (no headers)
16      8     u64   class_key
```

Sorted **descending** by `instance_count`. The number of entries is bounded by the number of classes.

Source: `src/class_index.rs`

---

## Dominator / retained heap files (optional)

Built only when indexing runs with retained-heap support enabled (the default for `hprof-toolkit index`). All four are written by the same step and are either all present or all absent; `HeapQuery::has_retained_heap()` reports which.

### dominators.bin

One entry per **reachable** object, mapping it to its immediate dominator.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   object_id     sort key
8       8     u64   dominator_id  immediate dominator; 0 = the virtual GC root
```

Sorted ascending by `object_id`.

### retained.bin

One entry per reachable object: the bytes freed if the object were collected.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   object_id       sort key
8       8     u64   retained_bytes  shallow size + retained sizes of everything it dominates
```

Sorted ascending by `object_id`. Shallow size counts instance field bytes and array element bytes only (no object header).

### retained_by_size.bin

`retained.bin` re-keyed by size so the largest retained subtrees are a prefix read.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   retained_bytes  sort key
8       8     u64   object_id
```

Sorted **descending** by `retained_bytes`.

### dominator_children.bin

The dominator tree keyed by parent: the children of `X` are the entries with `dominator_id == X`, found by binary search.

**Entry size:** 16 bytes

```
offset  size  type  field
0       8     u64   dominator_id  sort key; 0 = the virtual GC root (its children are the GC roots)
8       8     u64   object_id     secondary sort key
```

Sorted ascending by `(dominator_id, object_id)`.

Source: `src/dominator.rs`

---

## Array size index files

One file per Java array element type. Each entry represents one array, sorted so that the largest arrays appear first.

**Files:**

| File                 | Element type | Bytes per element        |
|----------------------|--------------|--------------------------|
| `array_boolean.bin`  | `boolean[]`  | 1                        |
| `array_char.bin`     | `char[]`     | 2                        |
| `array_float.bin`    | `float[]`    | 4                        |
| `array_double.bin`   | `double[]`   | 8                        |
| `array_byte.bin`     | `byte[]`     | 1                        |
| `array_short.bin`    | `short[]`    | 2                        |
| `array_int.bin`      | `int[]`      | 4                        |
| `array_long.bin`     | `long[]`     | 8                        |
| `array_object.bin`   | `Object[]`   | `id_size` (4 or 8 bytes) |

**Entry size:** 24 bytes

```
offset  size  type  field
0       8     u64   object_id   array object ID
8       8     u64   position    byte offset of the subtag byte in the hprof file
16      8     u64   byte_size   total bytes occupied by the array's elements (sort key)
```

Sorted **descending** by `byte_size` (largest arrays first), enabling O(1) access to the top-N largest arrays.

Source: `src/array_index.rs`

---

## Diff files (`{stem1}_vs_{stem2}.diff_indexes/`)

Built by `hprof-toolkit diff` (and on demand by `serve` and `mcp`) from two dumps of the same JVM, into a directory next to the first dump. Three entries, all sorted ascending by `object_id`, holding only *object* records (class dumps, instances, object and primitive arrays; not GC roots). The store is separate from the per-dump index directories and has no manifest: an existing set of the three files is reused as is, so delete the directory when either dump changes.

### removed.bin and added.bin

Objects present only in the first dump (`removed.bin`) or only in the second (`added.bin`).

**Entry size:** 24 bytes

```
offset  size  type      field
0       1     u8        tag        heap sub-record type
1       7     [u8; 7]   padding    zeros
8       8     u64       object_id  sort key
16      8     u64       position   byte offset of the sub-record in the respective hprof
```

### common.bin

Objects present in both dumps.

**Entry size:** 32 bytes

```
offset  size  type      field
0       1     u8        tag
1       1     u8        changed    0 = raw record bytes identical, 1 = differ
2       6     [u8; 6]   padding    zeros
8       8     u64       object_id  sort key
16      8     u64       position1  byte offset in the first hprof
24      8     u64       position2  byte offset in the second hprof
```

"Changed" compares the field bytes of instances, the element bytes of arrays and the whole record of classes. Object ids are heap addresses, so an id in `common.bin` may be a different object that reused the address; compare types on both sides when it matters.

Source: `src/diff_index.rs`, `src/diff.rs`
