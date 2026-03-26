# Index File Formats

All index files are written to `{hprof_stem}.indexes/` alongside the source `.hprof` file.
They are built in pipeline phases and memory-mapped at query time.

**Common properties across all files:**
- No file header, magic bytes, or version field — raw records start at offset 0
- All multi-byte integers are **little-endian**
- All files contain fixed-size records; entry count = `file_size / entry_size`
- Files are validated on open: `file_size % entry_size == 0`

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

## heap_index/ (per-segment files)

Phase 2 output. One directory per `HEAP_DUMP_SEGMENT` record, named by the segment's file offset. Each file inside holds one entry per heap sub-record within that segment.

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
