// Copyright 2024 Whippet Sort
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/testing/random.h>
#include <benchmark/benchmark.h>
#include <parquet/benchmark_util.h>
#include <parquet/column_reader.h>
#include <parquet/column_writer.h>
#include <parquet/encoding.h>
#include <parquet/file_reader.h>
#include <parquet/metadata.h>
#include <parquet/platform.h>
#include <parquet/schema.h>
#include <parquet/types.h>

using arrow::default_memory_pool;

namespace {

// The min/max number of values used to drive each family of encoding benchmarks
constexpr int MIN_RANGE = 4096;
constexpr int MAX_RANGE = 65536;
// Seed value used by random generator
constexpr int SEED = 1337;
// Cardinality Setting, 0 means totally random
constexpr int CARD_NARROW = 10;
constexpr int CARD_MEDIUM = 200;
constexpr int CARD_WIDE = 0;
}  // namespace

namespace parquet {

using schema::PrimitiveNode;

/************************Helper functions***************************/

// Used by File Writing tests
template <typename WriterType>
std::shared_ptr<WriterType> BuildWriter(int64_t output_size, const std::shared_ptr<ArrowOutputStream> &dst,
                                        ColumnChunkMetaDataBuilder *metadata, ColumnDescriptor *schema,
                                        const WriterProperties *properties, Compression::type codec) {
  std::unique_ptr<PageWriter> pager = PageWriter::Open(dst, codec, metadata);
  std::shared_ptr<ColumnWriter> writer = ColumnWriter::Make(metadata, std::move(pager), properties);
  return std::static_pointer_cast<WriterType>(writer);
}

// These Schemas are used in File IO Tests
std::shared_ptr<ColumnDescriptor> Int64Schema(Repetition::type repetition) {
  auto node = PrimitiveNode::Make("int64", repetition, Type::INT64);
  return std::make_shared<ColumnDescriptor>(node, repetition != Repetition::REQUIRED,
                                            repetition == Repetition::REPEATED);
}

std::shared_ptr<ColumnDescriptor> DoubleSchema(Repetition::type repetition) {
  auto node = PrimitiveNode::Make("double", repetition, Type::DOUBLE);
  return std::make_shared<ColumnDescriptor>(node, repetition != Repetition::REQUIRED,
                                            repetition == Repetition::REPEATED);
}

std::shared_ptr<ColumnDescriptor> ByteArraySchema(Repetition::type repetition) {
  auto node = PrimitiveNode::Make("byte_array", repetition, Type::BYTE_ARRAY);
  return std::make_shared<ColumnDescriptor>(node, repetition != Repetition::REQUIRED,
                                            repetition == Repetition::REPEATED);
}

// Given a vector of values, apply cardinality to a specific size
// This function assumes input values are unique.
template <typename T>
static auto ApplyCardinality(int cardinality, std::vector<T> values, int size) {
  if (cardinality == 0 || cardinality >= size) {
    return values;
  }
  auto result = std::vector<T>(size);
  for (int i = 0; i < size; i++) {
    // Potential problem: Reference copy for ByteArray
    result[i] = values[i % cardinality];
  }
  // Shuffle
  std::random_device rd;
  std::shuffle(result.begin(), result.end(), std::default_random_engine(rd()));
  return result;
}

// Helper functions for Dictionary Encoding
template <typename Type>
static void EncodeDict(const std::vector<typename Type::c_type> &values, ::benchmark::State &state,
                       std::shared_ptr<ColumnDescriptor> descr) {
  using T = typename Type::c_type;
  int num_values = static_cast<int>(values.size());

  MemoryPool *allocator = default_memory_pool();

  // Note: Encoding input is not used when use_dictionary is set to true
  auto base_encoder = MakeEncoder(Type::type_num, Encoding::RLE_DICTIONARY,
                                  /*use_dictionary=*/true, descr.get(), allocator);
  auto encoder = dynamic_cast<typename EncodingTraits<Type>::Encoder *>(base_encoder.get());
  for (auto _ : state) {
    encoder->Put(values.data(), num_values);
    encoder->FlushValues();
  }

  state.SetBytesProcessed(state.iterations() * num_values * sizeof(T));
  state.SetItemsProcessed(state.iterations() * num_values);
}

template <typename Type>
static void DecodeDict(const std::vector<typename Type::c_type> &values, ::benchmark::State &state,
                       std::shared_ptr<ColumnDescriptor> descr) {
  typedef typename Type::c_type T;
  int num_values = static_cast<int>(values.size());

  MemoryPool *allocator = default_memory_pool();

  auto base_encoder = MakeEncoder(Type::type_num, Encoding::PLAIN, true, descr.get(), allocator);
  auto encoder = dynamic_cast<typename EncodingTraits<Type>::Encoder *>(base_encoder.get());
  auto dict_traits = dynamic_cast<DictEncoder<Type> *>(base_encoder.get());
  encoder->Put(values.data(), num_values);

  std::shared_ptr<ResizableBuffer> dict_buffer = AllocateBuffer(allocator, dict_traits->dict_encoded_size());

  std::shared_ptr<ResizableBuffer> indices = AllocateBuffer(allocator, encoder->EstimatedDataEncodedSize());

  dict_traits->WriteDict(dict_buffer->mutable_data());
  int actual_bytes = dict_traits->WriteIndices(indices->mutable_data(), static_cast<int>(indices->size()));

  PARQUET_THROW_NOT_OK(indices->Resize(actual_bytes));

  std::vector<T> decoded_values(num_values);
  for (auto _ : state) {
    auto dict_decoder = MakeTypedDecoder<Type>(Encoding::PLAIN, descr.get());
    dict_decoder->SetData(dict_traits->num_entries(), dict_buffer->data(), static_cast<int>(dict_buffer->size()));

    auto decoder = MakeDictDecoder<Type>(descr.get());
    decoder->SetDict(dict_decoder.get());
    decoder->SetData(num_values, indices->data(), static_cast<int>(indices->size()));
    decoder->Decode(decoded_values.data(), num_values);
  }

  state.SetBytesProcessed(state.iterations() * num_values * sizeof(T));
  state.SetItemsProcessed(state.iterations() * num_values);
}

// This function is used to generate random Int64 values.
// Different Generator may be used to benefit Delta Binary Packing Encoding
static auto MakeInt64InputScatter(size_t length, int cardinality) {
  std::vector<uint8_t> heap;
  if (cardinality == 0) {
    // Random generation
    std::vector<int64_t> values(length);
    benchmark::GenerateBenchmarkData(length, SEED, values.data(), &heap, sizeof(int64_t));
    return values;
  } else {
    // Generate with cardinality
    std::vector<int64_t> values(cardinality);
    benchmark::GenerateBenchmarkData(cardinality, SEED, values.data(), &heap, sizeof(int64_t));
    return ApplyCardinality(cardinality, values, length);
  }
}
// This function is used to generate random Double values.
static auto MakeDoubleInput(size_t length, int cardinality) {
  std::vector<uint8_t> heap;
  if (cardinality == 0) {
    // Random generation
    std::vector<double> values(length);
    benchmark::GenerateBenchmarkData(length, SEED, values.data(), &heap, 8);
    return values;
  } else {
    // Generate with cardinality
    std::vector<double> values(cardinality);
    benchmark::GenerateBenchmarkData(cardinality, SEED, values.data(), &heap, 8);
    return ApplyCardinality(cardinality, values, length);
  }
}

/*******************Encoding Decoding Tests*******************/
// Int64 Plain Encode Test
// NumberGenerator: Function that can produce input values
template <typename NumberGenerator>
static void BM_PlainEncodingInt64(::benchmark::State &state, NumberGenerator gen, int cardinality) {
  std::vector<int64_t> values = gen(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<Int64Type>(Encoding::PLAIN);
  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    encoder->FlushValues();
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(int64_t));
}

static void BM_PlainEncodingInt64_Narrow(::benchmark::State &state) {
  BM_PlainEncodingInt64(state, MakeInt64InputScatter, CARD_NARROW);
}

static void BM_PlainEncodingInt64_Medium(::benchmark::State &state) {
  BM_PlainEncodingInt64(state, MakeInt64InputScatter, CARD_MEDIUM);
}

static void BM_PlainEncodingInt64_Wide(::benchmark::State &state) {
  BM_PlainEncodingInt64(state, MakeInt64InputScatter, CARD_WIDE);
}

BENCHMARK(BM_PlainEncodingInt64_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainEncodingInt64_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainEncodingInt64_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Int64 Decode Test
template <typename NumberGenerator>
static void BM_PlainDecodingInt64(::benchmark::State &state, NumberGenerator gen, int cardinality) {
  std::vector<int64_t> values = gen(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<Int64Type>(Encoding::PLAIN);
  encoder->Put(values.data(), static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buf = encoder->FlushValues();
  auto decoder = MakeTypedDecoder<Int64Type>(Encoding::PLAIN);
  for (auto _ : state) {
    decoder->SetData(static_cast<int>(values.size()), buf->data(), static_cast<int>(buf->size()));
    decoder->Decode(values.data(), static_cast<int>(values.size()));
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(int64_t));
}

static void BM_PlainDecodingInt64_Narrow(::benchmark::State &state) {
  BM_PlainDecodingInt64(state, MakeInt64InputScatter, CARD_NARROW);
}

static void BM_PlainDecodingInt64_Medium(::benchmark::State &state) {
  BM_PlainDecodingInt64(state, MakeInt64InputScatter, CARD_MEDIUM);
}

static void BM_PlainDecodingInt64_Wide(::benchmark::State &state) {
  BM_PlainDecodingInt64(state, MakeInt64InputScatter, CARD_WIDE);
}
BENCHMARK(BM_PlainDecodingInt64_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainDecodingInt64_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainDecodingInt64_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Int64 Delta Encode Test
template <typename NumberGenerator>
static void BM_DeltaBitPackingEncode(::benchmark::State &state, NumberGenerator gen, int cardinality) {
  std::vector<int64_t> values = gen(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<Int64Type>(Encoding::DELTA_BINARY_PACKED);
  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    encoder->FlushValues();
  }
  state.SetBytesProcessed(state.iterations() * values.size() * sizeof(int64_t));
  state.SetItemsProcessed(state.iterations() * values.size());
}

static void BM_DeltaBitPackingEncode_Narrow(::benchmark::State &state) {
  BM_DeltaBitPackingEncode(state, MakeInt64InputScatter, CARD_NARROW);
}

static void BM_DeltaBitPackingEncode_Medium(::benchmark::State &state) {
  BM_DeltaBitPackingEncode(state, MakeInt64InputScatter, CARD_MEDIUM);
}

static void BM_DeltaBitPackingEncode_Wide(::benchmark::State &state) {
  BM_DeltaBitPackingEncode(state, MakeInt64InputScatter, CARD_WIDE);
}

BENCHMARK(BM_DeltaBitPackingEncode_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DeltaBitPackingEncode_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DeltaBitPackingEncode_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Int64 Delta Test Decode Test
template <typename NumberGenerator>
static void BM_DeltaBitPackingDecode(::benchmark::State &state, NumberGenerator gen, int cardinality) {
  std::vector<int64_t> values = gen(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<Int64Type>(Encoding::DELTA_BINARY_PACKED);
  encoder->Put(values.data(), static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buf = encoder->FlushValues();

  auto decoder = MakeTypedDecoder<Int64Type>(Encoding::DELTA_BINARY_PACKED);
  for (auto _ : state) {
    decoder->SetData(static_cast<int>(values.size()), buf->data(), static_cast<int>(buf->size()));
    decoder->Decode(values.data(), static_cast<int>(values.size()));
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(int64_t));
  state.SetItemsProcessed(state.iterations() * state.range(0));
}
static void BM_DeltaBitPackingDecode_Narrow(::benchmark::State &state) {
  BM_DeltaBitPackingDecode(state, MakeInt64InputScatter, CARD_NARROW);
}

static void BM_DeltaBitPackingDecode_Medium(::benchmark::State &state) {
  BM_DeltaBitPackingDecode(state, MakeInt64InputScatter, CARD_MEDIUM);
}

static void BM_DeltaBitPackingDecode_Wide(::benchmark::State &state) {
  BM_DeltaBitPackingDecode(state, MakeInt64InputScatter, CARD_WIDE);
}

BENCHMARK(BM_DeltaBitPackingDecode_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DeltaBitPackingDecode_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DeltaBitPackingDecode_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Int64 Dictionary Encode Test
template <typename Type, typename NumberGenerator>
static void BM_DictEncoding(::benchmark::State &state, NumberGenerator gen, int cardinality,
                            std::shared_ptr<ColumnDescriptor> descr) {
  using T = typename Type::c_type;
  std::vector<T> values = gen(state.range(0), cardinality);
  EncodeDict<Type>(values, state, descr);
}

static void BM_DictEncodingInt64_Narrow(::benchmark::State &state) {
  BM_DictEncoding<Int64Type>(state, MakeInt64InputScatter, CARD_NARROW, Int64Schema(Repetition::REQUIRED));
}

static void BM_DictEncodingInt64_Medium(::benchmark::State &state) {
  BM_DictEncoding<Int64Type>(state, MakeInt64InputScatter, CARD_MEDIUM, Int64Schema(Repetition::REQUIRED));
}

static void BM_DictEncodingInt64_Wide(::benchmark::State &state) {
  BM_DictEncoding<Int64Type>(state, MakeInt64InputScatter, CARD_WIDE, Int64Schema(Repetition::REQUIRED));
}
BENCHMARK(BM_DictEncodingInt64_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictEncodingInt64_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictEncodingInt64_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Int64 Dictionary Decode Test
template <typename Type, typename NumberGenerator>
static void BM_DictDecoding(::benchmark::State &state, NumberGenerator gen, int cardinality,
                            std::shared_ptr<ColumnDescriptor> descr) {
  using T = typename Type::c_type;
  std::vector<T> values = gen(state.range(0), cardinality);
  DecodeDict<Type>(values, state, descr);
}

static void BM_DictDecodingInt64_Narrow(::benchmark::State &state) {
  BM_DictDecoding<Int64Type>(state, MakeInt64InputScatter, CARD_NARROW, Int64Schema(Repetition::REQUIRED));
}

static void BM_DictDecodingInt64_Medium(::benchmark::State &state) {
  BM_DictDecoding<Int64Type>(state, MakeInt64InputScatter, CARD_MEDIUM, Int64Schema(Repetition::REQUIRED));
}

static void BM_DictDecodingInt64_Wide(::benchmark::State &state) {
  BM_DictDecoding<Int64Type>(state, MakeInt64InputScatter, CARD_WIDE, Int64Schema(Repetition::REQUIRED));
}
BENCHMARK(BM_DictDecodingInt64_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictDecodingInt64_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictDecodingInt64_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Double Plain Encode Test
static void BM_PlainEncodingDouble(::benchmark::State &state, int cardinality) {
  std::vector<double> values = MakeDoubleInput(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<DoubleType>(Encoding::PLAIN);
  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    encoder->FlushValues();
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}

static void BM_PlainEncodingDouble_Narrow(::benchmark::State &state) { BM_PlainEncodingDouble(state, CARD_NARROW); }

static void BM_PlainEncodingDouble_Medium(::benchmark::State &state) { BM_PlainEncodingDouble(state, CARD_MEDIUM); }

static void BM_PlainEncodingDouble_Wide(::benchmark::State &state) { BM_PlainEncodingDouble(state, CARD_WIDE); }

BENCHMARK(BM_PlainEncodingDouble_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainEncodingDouble_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainEncodingDouble_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Double Plain Decode Test
static void BM_PlainDecodingDouble(::benchmark::State &state, int cardinality) {
  std::vector<double> values = MakeDoubleInput(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<DoubleType>(Encoding::PLAIN);
  encoder->Put(values.data(), static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buf = encoder->FlushValues();

  for (auto _ : state) {
    auto decoder = MakeTypedDecoder<DoubleType>(Encoding::PLAIN);
    decoder->SetData(static_cast<int>(values.size()), buf->data(), static_cast<int>(buf->size()));
    decoder->Decode(values.data(), static_cast<int>(values.size()));
  }
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}

static void BM_PlainDecodingDouble_Narrow(::benchmark::State &state) { BM_PlainDecodingDouble(state, CARD_NARROW); }

static void BM_PlainDecodingDouble_Medium(::benchmark::State &state) { BM_PlainDecodingDouble(state, CARD_MEDIUM); }

static void BM_PlainDecodingDouble_Wide(::benchmark::State &state) { BM_PlainDecodingDouble(state, CARD_WIDE); }

BENCHMARK(BM_PlainDecodingDouble_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainDecodingDouble_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_PlainDecodingDouble_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Double Stream Split Encode Test
// Note: There are multiple encoding functions for this encoding, thus
// encode_func parameter is used here, one should pass appropriate encoding
// function to it.
// ::arrow::util::internal::ByteStreamSplitEncode is used here. Similar approach
// is used in decoding test.
static void BM_ByteStreamSplitEncode(::benchmark::State &state, int cardinality) {
  std::vector<double> values = MakeDoubleInput(state.range(0), cardinality);
  auto encoder = MakeTypedEncoder<DoubleType>(Encoding::BYTE_STREAM_SPLIT);

  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    std::shared_ptr<Buffer> buf = encoder->FlushValues();
  }
  state.SetBytesProcessed(state.iterations() * values.size() * sizeof(double));
  state.SetItemsProcessed(state.iterations() * values.size());
}

static void BM_ByteStreamSplitEncode_Narrow(::benchmark::State &state) { BM_ByteStreamSplitEncode(state, CARD_NARROW); }

static void BM_ByteStreamSplitEncode_Medium(::benchmark::State &state) { BM_ByteStreamSplitEncode(state, CARD_MEDIUM); }

static void BM_ByteStreamSplitEncode_Wide(::benchmark::State &state) { BM_ByteStreamSplitEncode(state, CARD_WIDE); }

BENCHMARK(BM_ByteStreamSplitEncode_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_ByteStreamSplitEncode_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_ByteStreamSplitEncode_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Double Split Decode Test
static void BM_ByteStreamSplitDecode(::benchmark::State &state, int cardinality) {
  std::vector<double> values = MakeDoubleInput(state.range(0), cardinality);
  auto size = static_cast<int>(values.size());
  auto encoder = MakeTypedEncoder<DoubleType>(Encoding::BYTE_STREAM_SPLIT);
  encoder->Put(values.data(), size);
  auto buf = encoder->FlushValues();
  auto decoder = MakeTypedDecoder<DoubleType>(Encoding::BYTE_STREAM_SPLIT);

  for (auto _ : state) {
    decoder->SetData(size, buf->data(), buf->size());
    decoder->Decode(values.data(), size);
  }
  state.SetBytesProcessed(state.iterations() * values.size() * sizeof(double));
  state.SetItemsProcessed(state.iterations() * values.size());
}

static void BM_ByteStreamSplitDecode_Narrow(::benchmark::State &state) { BM_ByteStreamSplitDecode(state, CARD_NARROW); }

static void BM_ByteStreamSplitDecode_Medium(::benchmark::State &state) { BM_ByteStreamSplitDecode(state, CARD_MEDIUM); }

static void BM_ByteStreamSplitDecode_Wide(::benchmark::State &state) { BM_ByteStreamSplitDecode(state, CARD_WIDE); }

BENCHMARK(BM_ByteStreamSplitDecode_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_ByteStreamSplitDecode_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_ByteStreamSplitDecode_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Double Dictionary Encode Test
static void BM_DictEncodingDouble_Narrow(::benchmark::State &state) {
  BM_DictEncoding<DoubleType>(state, MakeDoubleInput, CARD_NARROW, DoubleSchema(Repetition::REQUIRED));
}

static void BM_DictEncodingDouble_Medium(::benchmark::State &state) {
  BM_DictEncoding<DoubleType>(state, MakeDoubleInput, CARD_MEDIUM, DoubleSchema(Repetition::REQUIRED));
}

static void BM_DictEncodingDouble_Wide(::benchmark::State &state) {
  BM_DictEncoding<DoubleType>(state, MakeDoubleInput, CARD_WIDE, DoubleSchema(Repetition::REQUIRED));
}

// Double Dict Decode Test
static void BM_DictDecodingDouble_Narrow(::benchmark::State &state) {
  BM_DictDecoding<DoubleType>(state, MakeDoubleInput, CARD_NARROW, DoubleSchema(Repetition::REQUIRED));
}

static void BM_DictDecodingDouble_Medium(::benchmark::State &state) {
  BM_DictDecoding<DoubleType>(state, MakeDoubleInput, CARD_MEDIUM, DoubleSchema(Repetition::REQUIRED));
}

static void BM_DictDecodingDouble_Wide(::benchmark::State &state) {
  BM_DictDecoding<DoubleType>(state, MakeDoubleInput, CARD_WIDE, DoubleSchema(Repetition::REQUIRED));
}
BENCHMARK(BM_DictEncodingDouble_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictEncodingDouble_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictEncodingDouble_Wide)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictDecodingDouble_Narrow)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictDecodingDouble_Medium)->Range(MIN_RANGE, MAX_RANGE);
BENCHMARK(BM_DictDecodingDouble_Wide)->Range(MIN_RANGE, MAX_RANGE);

// Byte Array Encoding Benchmarks
// This function is for Plain Encoding and Delta Encoding.
// Dictionary Encoding needs to handle separately since dictionary encoding
// might turn back to other encoding when page becomes too large. We need to
// explicitly use dictionary encoder.
void EncodingByteArrayBenchmark(::benchmark::State &state, Encoding::type encoding) {
  ::arrow::random::RandomArrayGenerator rag(SEED);
  // Using arrow generator to generate random data.
  int32_t min_length = static_cast<int32_t>(state.range(0));
  int32_t max_length = static_cast<int32_t>(state.range(1));
  int32_t array_size = static_cast<int32_t>(state.range(2));
  auto array = rag.String(/* size */ array_size, /* min_length */ min_length,
                          /* max_length */ max_length,
                          /* null_probability */ 0);
  const auto array_actual = ::arrow::internal::checked_pointer_cast<::arrow::StringArray>(array);
  auto encoder = MakeTypedEncoder<ByteArrayType>(encoding);
  std::vector<ByteArray> values;
  for (int i = 0; i < array_actual->length(); ++i) {
    values.emplace_back(array_actual->GetView(i));
  }

  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    encoder->FlushValues();
  }
  state.SetItemsProcessed(state.iterations() * array_actual->length());
  state.SetBytesProcessed(state.iterations() *
                          (array_actual->value_data()->size() + array_actual->value_offsets()->size()));
}

static void BM_DeltaLengthEncodingByteArray(::benchmark::State &state) {
  EncodingByteArrayBenchmark(state, Encoding::DELTA_LENGTH_BYTE_ARRAY);
}

static void BM_PlainEncodingByteArray(::benchmark::State &state) { EncodingByteArrayBenchmark(state, Encoding::PLAIN); }

// For Dictionary Encoding
static void BM_DictEncodingByteArray(::benchmark::State &state) {
  ::arrow::random::RandomArrayGenerator rag(SEED);
  // Using arrow generator to generate random data.
  int32_t min_length = static_cast<int32_t>(state.range(0));
  int32_t max_length = static_cast<int32_t>(state.range(1));
  int32_t array_size = static_cast<int32_t>(state.range(2));
  auto array = rag.String(/* size */ array_size, /* min_length */ min_length,
                          /* max_length */ max_length,
                          /* null_probability */ 0);
  const auto array_actual = ::arrow::internal::checked_pointer_cast<::arrow::StringArray>(array);
  auto encoder = MakeDictDecoder<ByteArrayType>();
  std::vector<ByteArray> values;
  for (int i = 0; i < array_actual->length(); ++i) {
    values.emplace_back(array_actual->GetView(i));
  }
  EncodeDict<ByteArrayType>(values, state, ByteArraySchema(Repetition::REQUIRED));
  state.SetItemsProcessed(state.iterations() * array_actual->length());
  state.SetBytesProcessed(state.iterations() *
                          (array_actual->value_data()->size() + array_actual->value_offsets()->size()));
}

BENCHMARK(BM_DictEncodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DictEncodingByteArray)->Args({10, 1024, MAX_RANGE});
BENCHMARK(BM_PlainEncodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DeltaLengthEncodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DeltaLengthEncodingByteArray)->Args({100, 200, MAX_RANGE});

// Byte Array Decoding Benchmarks
void DecodingByteArrayBenchmark(::benchmark::State &state, Encoding::type encoding) {
  ::arrow::random::RandomArrayGenerator rag(SEED);
  int32_t min_length = static_cast<int32_t>(state.range(0));
  int32_t max_length = static_cast<int32_t>(state.range(1));
  int32_t array_size = static_cast<int32_t>(state.range(2));
  // Using arrow to write, because we just benchmark decoding here.
  auto array = rag.String(/* size */ array_size, /* min_length */ min_length,
                          /* max_length */ max_length,
                          /* null_probability */ 0);
  const auto array_actual = ::arrow::internal::checked_pointer_cast<::arrow::StringArray>(array);
  auto encoder = MakeTypedEncoder<ByteArrayType>(encoding);
  encoder->Put(*array);
  std::shared_ptr<Buffer> buf = encoder->FlushValues();

  std::vector<ByteArray> values;
  values.resize(array->length());
  for (auto _ : state) {
    auto decoder = MakeTypedDecoder<ByteArrayType>(encoding);
    decoder->SetData(static_cast<int>(array->length()), buf->data(), static_cast<int>(buf->size()));
    decoder->Decode(values.data(), static_cast<int>(values.size()));
    ::benchmark::DoNotOptimize(values);
  }
  state.SetItemsProcessed(state.iterations() * array->length());
  state.SetBytesProcessed(state.iterations() *
                          (array_actual->value_data()->size() + array_actual->value_offsets()->size()));
}

static void BM_PlainDecodingByteArray(::benchmark::State &state) { DecodingByteArrayBenchmark(state, Encoding::PLAIN); }

static void BM_DeltaLengthDecodingByteArray(::benchmark::State &state) {
  DecodingByteArrayBenchmark(state, Encoding::DELTA_LENGTH_BYTE_ARRAY);
}

static void BM_DictDecodingByteArray(::benchmark::State &state) {
  ::arrow::random::RandomArrayGenerator rag(SEED);
  // Using arrow generator to generate random data.
  int32_t min_length = static_cast<int32_t>(state.range(0));
  int32_t max_length = static_cast<int32_t>(state.range(1));
  int32_t array_size = static_cast<int32_t>(state.range(2));
  auto array = rag.String(/* size */ array_size, /* min_length */ min_length,
                          /* max_length */ max_length,
                          /* null_probability */ 0);
  const auto array_actual = ::arrow::internal::checked_pointer_cast<::arrow::StringArray>(array);
  auto encoder = MakeDictDecoder<ByteArrayType>();
  std::vector<ByteArray> values;
  for (int i = 0; i < array_actual->length(); ++i) {
    values.emplace_back(array_actual->GetView(i));
  }
  DecodeDict<ByteArrayType>(values, state, ByteArraySchema(Repetition::REQUIRED));
  ;

  state.SetItemsProcessed(state.iterations() * array_actual->length());
  state.SetBytesProcessed(state.iterations() *
                          (array_actual->value_data()->size() + array_actual->value_offsets()->size()));
}

BENCHMARK(BM_PlainDecodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DeltaLengthDecodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DeltaLengthDecodingByteArray)->Args({10, 1024, MAX_RANGE});
BENCHMARK(BM_DictDecodingByteArray)->Args({10, 20, MAX_RANGE});
BENCHMARK(BM_DictDecodingByteArray)->Args({10, 1024, MAX_RANGE});

// For Delta Encoding
// This function is the same as the one in encoding benchmark
struct DeltaByteArrayState {
  int32_t min_size = 0;
  int32_t max_size;
  int32_t array_length;
  int32_t total_data_size = 0;
  double prefixed_probability;
  std::vector<uint8_t> buf;

  explicit DeltaByteArrayState(const ::benchmark::State &state)
      : max_size(static_cast<int32_t>(state.range(0))),
        array_length(static_cast<int32_t>(state.range(1))),
        prefixed_probability(state.range(2) / 100.0) {}

  std::vector<ByteArray> MakeRandomByteArray(uint32_t seed) {
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> dist_size(min_size, max_size);
    std::uniform_int_distribution<int> dist_byte(0, 255);
    std::bernoulli_distribution dist_has_prefix(prefixed_probability);
    std::uniform_real_distribution<double> dist_prefix_length(0, 1);

    std::vector<ByteArray> out(array_length);
    buf.resize(max_size * array_length);
    auto buf_ptr = buf.data();
    total_data_size = 0;

    for (int32_t i = 0; i < array_length; ++i) {
      int len = dist_size(gen);
      out[i].len = len;
      out[i].ptr = buf_ptr;

      bool do_prefix = i > 0 && dist_has_prefix(gen);
      int prefix_len = 0;
      if (do_prefix) {
        int max_prefix_len = std::min(len, static_cast<int>(out[i - 1].len));
        prefix_len = static_cast<int>(std::ceil(max_prefix_len * dist_prefix_length(gen)));
      }
      for (int j = 0; j < prefix_len; ++j) {
        buf_ptr[j] = out[i - 1].ptr[j];
      }
      for (int j = prefix_len; j < len; ++j) {
        buf_ptr[j] = static_cast<uint8_t>(dist_byte(gen));
      }
      buf_ptr += len;
      total_data_size += len;
    }
    return out;
  }
};

static void BM_DeltaEncodingByteArray(::benchmark::State &state) {
  DeltaByteArrayState delta_state(state);
  std::vector<ByteArray> values = delta_state.MakeRandomByteArray(/*seed=*/42);

  auto encoder = MakeTypedEncoder<ByteArrayType>(Encoding::DELTA_BYTE_ARRAY);
  const int64_t plain_encoded_size = delta_state.total_data_size + 4 * delta_state.array_length;
  int64_t encoded_size = 0;

  for (auto _ : state) {
    encoder->Put(values.data(), static_cast<int>(values.size()));
    encoded_size = encoder->FlushValues()->size();
  }
  state.SetItemsProcessed(state.iterations() * delta_state.array_length);
  state.SetBytesProcessed(state.iterations() * delta_state.total_data_size);
  state.counters["compression_ratio"] = static_cast<double>(plain_encoded_size) / encoded_size;
}

static void BM_DeltaDecodingByteArray(::benchmark::State &state) {
  DeltaByteArrayState delta_state(state);
  std::vector<ByteArray> values = delta_state.MakeRandomByteArray(/*seed=*/42);

  auto encoder = MakeTypedEncoder<ByteArrayType>(Encoding::DELTA_BYTE_ARRAY);
  encoder->Put(values.data(), static_cast<int>(values.size()));
  std::shared_ptr<Buffer> buf = encoder->FlushValues();

  const int64_t plain_encoded_size = delta_state.total_data_size + 4 * delta_state.array_length;
  const int64_t encoded_size = buf->size();

  auto decoder = MakeTypedDecoder<ByteArrayType>(Encoding::DELTA_BYTE_ARRAY);
  for (auto _ : state) {
    decoder->SetData(delta_state.array_length, buf->data(), static_cast<int>(buf->size()));
    decoder->Decode(values.data(), static_cast<int>(values.size()));
    ::benchmark::DoNotOptimize(values);
  }
  state.SetItemsProcessed(state.iterations() * delta_state.array_length);
  state.SetBytesProcessed(state.iterations() * delta_state.total_data_size);
  state.counters["compression_ratio"] = static_cast<double>(plain_encoded_size) / encoded_size;
}

static void ByteArrayDeltaCustomArguments(::benchmark::internal::Benchmark *b) {
  for (int max_string_length : {8, 64, 1024}) {
    for (int batch_size : {512, 2048}) {
      for (int prefixed_percent : {10, 90, 99}) {
        b->Args({max_string_length, batch_size, prefixed_percent});
      }
    }
  }
  b->ArgNames({"max-string-length", "batch-size", "prefixed-percent"});
}

BENCHMARK(BM_DeltaEncodingByteArray)->Apply(ByteArrayDeltaCustomArguments);
BENCHMARK(BM_DeltaDecodingByteArray)->Apply(ByteArrayDeltaCustomArguments);
/**************************Reading/Writing
   Tests**************************************/

// Int64 Writing Tests
// Currently only REQUIRED is tested.
void SetBytesProcessed(::benchmark::State &state, Repetition::type repetition) {
  int64_t num_values = state.iterations() * state.range(0);
  int64_t bytes_processed = num_values * sizeof(int64_t);
  if (repetition != Repetition::REQUIRED) {
    bytes_processed += num_values * sizeof(int16_t);
  }
  if (repetition == Repetition::REPEATED) {
    bytes_processed += num_values * sizeof(int16_t);
  }
  state.SetBytesProcessed(bytes_processed);
  state.SetItemsProcessed(num_values);
}

template <typename WriterType, typename Type, typename NumberGenerator>
static void BM_WriteColumn(::benchmark::State &state, Compression::type codec, Encoding::type encoding, int cardinality,
                           bool use_dict, NumberGenerator gen, std::shared_ptr<parquet::ColumnDescriptor> schema) {
  using T = typename Type::c_type;
  auto input_values = gen(state.range(0), cardinality);

  std::vector<int16_t> definition_levels(state.range(0), 1);
  std::vector<int16_t> repetition_levels(state.range(0), 0);
  std::shared_ptr<WriterProperties> properties;
  if (use_dict) {
    properties = WriterProperties::Builder().compression(codec)->encoding(encoding)->build();
  } else {
    properties = WriterProperties::Builder().compression(codec)->encoding(encoding)->disable_dictionary()->build();
  }
  auto metadata = ColumnChunkMetaDataBuilder::Make(properties, schema.get());

  int64_t data_size = input_values.size() * sizeof(T);
  int64_t stream_size = 0;
  for (auto _ : state) {
    // Clear the filesystem cache (requires root access)
    system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null");
    auto start = std::chrono::high_resolution_clock::now();
    auto stream = CreateOutputStream();
    std::shared_ptr<WriterType> writer =
        BuildWriter<WriterType>(state.range(0), stream, metadata.get(), schema.get(), properties.get(), codec);
    writer->WriteBatch(input_values.size(), definition_levels.data(), repetition_levels.data(), input_values.data());
    writer->Close();
    stream_size = stream->Tell().ValueOrDie();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
  SetBytesProcessed(state, Repetition::REQUIRED);
  state.counters["compression_ratio"] = static_cast<double>(data_size) / stream_size;
}

template <const int cardinality, Encoding::type encoding, bool use_dict>
static void BM_WriteInt64Column(::benchmark::State &state) {
  BM_WriteColumn<Int64Writer, Int64Type>(state, Compression::UNCOMPRESSED, encoding, cardinality, use_dict,
                                         MakeInt64InputScatter, Int64Schema(Repetition::REQUIRED));
}

// Int64 Plain  Write Test
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_NARROW, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_MEDIUM, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_WIDE, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();
// Int64 Delta Write Tests
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_NARROW, Encoding::DELTA_BINARY_PACKED, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_MEDIUM, Encoding::DELTA_BINARY_PACKED, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_WIDE, Encoding::DELTA_BINARY_PACKED, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();
// Int64 Dictionary Write Tests
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_NARROW, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_MEDIUM, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();
BENCHMARK_TEMPLATE(BM_WriteInt64Column, CARD_WIDE, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();

// Int64 Reading Test

// Used by File Reading tests
template <typename ReadType>
std::shared_ptr<ReadType> BuildReader(std::shared_ptr<Buffer> &buffer, int64_t num_values, Compression::type codec,
                                      ColumnDescriptor *schema) {
  auto source = std::make_shared<::arrow::io::BufferReader>(buffer);
  std::unique_ptr<PageReader> page_reader = PageReader::Open(source, num_values, codec);
  return std::static_pointer_cast<ReadType>(ColumnReader::Make(schema, std::move(page_reader)));
}

template <typename WriterType, typename ReaderType, typename Type, typename NumberGenerator>
static void BM_ReadColumn(::benchmark::State &state, Compression::type codec, Encoding::type encoding,
                          NumberGenerator gen, int cardinality, bool use_dict,
                          std::shared_ptr<parquet::ColumnDescriptor> schema) {
  using T = typename Type::c_type;

  const auto &input_values = gen(state.range(0), cardinality);

  std::vector<int16_t> definition_levels(state.range(0), 1);
  std::vector<int16_t> repetition_levels(state.range(0), 0);
  std::shared_ptr<WriterProperties> properties;
  if (use_dict) {
    // Dictionary is enabled by default
    properties = WriterProperties::Builder().compression(codec)->encoding(encoding)->build();
  } else {
    properties = WriterProperties::Builder().compression(codec)->encoding(encoding)->disable_dictionary()->build();
  }

  auto metadata = ColumnChunkMetaDataBuilder::Make(properties, schema.get());

  auto stream = CreateOutputStream();
  std::shared_ptr<WriterType> writer =
      BuildWriter<WriterType>(state.range(0), stream, metadata.get(), schema.get(), properties.get(), codec);
  writer->WriteBatch(input_values.size(), definition_levels.data(), repetition_levels.data(), input_values.data());
  writer->Close();

  PARQUET_ASSIGN_OR_THROW(auto src, stream->Finish());
  int64_t stream_size = src->size();
  int64_t data_size = input_values.size() * sizeof(T);

  std::vector<T> values_out(state.range(1));
  std::vector<int16_t> definition_levels_out(state.range(1));
  std::vector<int16_t> repetition_levels_out(state.range(1));
  for (auto _ : state) {
    // Drop cache
    system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null");
    auto start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<ReaderType> reader = BuildReader<ReaderType>(src, state.range(1), codec, schema.get());
    int64_t values_read = 0;
    for (size_t i = 0; i < input_values.size(); i += values_read) {
      reader->ReadBatch(values_out.size(), definition_levels_out.data(), repetition_levels_out.data(),
                        values_out.data(), &values_read);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
  SetBytesProcessed(state, Repetition::REQUIRED);
  state.counters["compression_ratio"] = static_cast<double>(data_size) / stream_size;
}

template <const int cardinality, Encoding::type encoding, bool use_dict>
static void BM_ReadInt64Column(::benchmark::State &state) {
  BM_ReadColumn<Int64Writer, Int64Reader, Int64Type>(state, Compression::UNCOMPRESSED, encoding, MakeInt64InputScatter,
                                                     cardinality, use_dict, Int64Schema(Repetition::REQUIRED));
}

// Int64 Plain Read: First argument is the input size and the second argument is
// read size
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_NARROW, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_MEDIUM, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_WIDE, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
// Int64 Delta Read
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_NARROW, Encoding::DELTA_BINARY_PACKED, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_MEDIUM, Encoding::DELTA_BINARY_PACKED, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_WIDE, Encoding::DELTA_BINARY_PACKED, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
// Int64 Dictionary Read
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_NARROW, Encoding::PLAIN, true)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_MEDIUM, Encoding::PLAIN, true)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadInt64Column, CARD_WIDE, Encoding::PLAIN, true)->Args({MAX_RANGE, MAX_RANGE})->UseManualTime();

// Double Writing test
template <const int cardinality, Encoding::type encoding, bool use_dict>
static void BM_WriteDoubleColumn(::benchmark::State &state) {
  BM_WriteColumn<DoubleWriter, DoubleType>(state, Compression::UNCOMPRESSED, encoding, cardinality, use_dict,
                                           MakeDoubleInput, DoubleSchema(Repetition::REQUIRED));
}

// Double Plain Write Test
BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_NARROW, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_MEDIUM, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_WIDE, Encoding::PLAIN, false)->Arg(MAX_RANGE)->UseManualTime();

// Double Split Stream Write Test
BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_NARROW, Encoding::BYTE_STREAM_SPLIT, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_MEDIUM, Encoding::BYTE_STREAM_SPLIT, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_WIDE, Encoding::BYTE_STREAM_SPLIT, false)
    ->Arg(MAX_RANGE)
    ->UseManualTime();

// Double Dictionary Write Test
BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_NARROW, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_MEDIUM, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();

BENCHMARK_TEMPLATE(BM_WriteDoubleColumn, CARD_WIDE, Encoding::PLAIN, true)->Arg(MAX_RANGE)->UseManualTime();

// Double Reading Test
template <const int cardinality, Encoding::type encoding, bool use_dict>
static void BM_ReadDoubleColumn(::benchmark::State &state) {
  BM_ReadColumn<DoubleWriter, DoubleReader, DoubleType>(state, Compression::UNCOMPRESSED, encoding, MakeDoubleInput,
                                                        cardinality, use_dict, DoubleSchema(Repetition::REQUIRED));
}

// Double Plain Read: First argument is the input size and the second argument
// is read size
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_NARROW, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_MEDIUM, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_WIDE, Encoding::PLAIN, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();

// Double BYTE_STREAM_SPLIT Read
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_NARROW, Encoding::BYTE_STREAM_SPLIT, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_MEDIUM, Encoding::BYTE_STREAM_SPLIT, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_WIDE, Encoding::BYTE_STREAM_SPLIT, false)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();

// Double Dictionary Read
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_NARROW, Encoding::PLAIN, true)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_MEDIUM, Encoding::PLAIN, true)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ReadDoubleColumn, CARD_WIDE, Encoding::PLAIN, true)
    ->Args({MAX_RANGE, MAX_RANGE})
    ->UseManualTime();

// TODO: Byte array read/write benchmarks. Needs to handle size problem
}  // namespace parquet

BENCHMARK_MAIN();
