#include "parquet_sorter.h"

#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/compute/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <queue>
#include <set>

#include "parquet/column_page.h"
#include "parquet/column_reader.h"
#include "parquet/encoding.h"
#include "parquet/file_reader.h"

using std::deque;
using std::future;
using std::pair;
namespace whippet_sort {

string ParquetPageTypeToString(parquet::PageType::type type) {
  switch (type) {
    case parquet::PageType::DATA_PAGE:
      return "DATA_PAGE";
    case parquet::PageType::DATA_PAGE_V2:
      return "DATA_PAGE_V2";
    case parquet::PageType::DICTIONARY_PAGE:
      return "DICTIONARY_PAGE";
    case parquet::PageType::INDEX_PAGE:
      return "INDEX_PAGE";
    case parquet::PageType::UNDEFINED:
      return "UNDEFINED";
    default:
      return "UNSUPPORTED PAGE TYPE";
  }
}

void CountBaseSort::set_global_index_for_chunk(
    IndexType* offset, int rg_idx, uint32_t col_idx,
    vector<IndexType>& global_index, shared_ptr<parquet::PageReader> reader) {
  // Iterate through pages to place the position into the result
  // Skip first page
  IndexType current_index = this->index_offset[rg_idx];
  shared_ptr<parquet::Page> page = reader->NextPage();
  vector<int32_t> values;  // Temp container for index decoding
  auto index_decoder = parquet::MakeDictDecoder<parquet::Int64Type>();
  while (page = reader->NextPage(), page != nullptr) {
    if (page->type() == parquet::PageType::DATA_PAGE) {
      auto index_page = std::static_pointer_cast<parquet::DataPage>(page);
      index_decoder->SetData(static_cast<int>(index_page->num_values()),
                             index_page->data(), index_page->size());
      values.resize(index_page->num_values());
      index_decoder->DecodeIndices(index_page->num_values(), values.data());
      for (auto& dict_idx : values) {
        global_index[offset[dict_idx]] = current_index;
        offset[dict_idx]++;
        current_index++;
      }
    } else if (page->type() == parquet::PageType::INDEX_PAGE) {
      // Skip Index page for now. Can be used to optimize
      continue;
    } else {
      string page_type = ParquetPageTypeToString(page->type());
      throw std::runtime_error(
          "Unexpected page type. Expecting: IndexPage. Met: " + page_type);
    }
  }  // End of page iteration within Chunk
}

vector<IndexType> CountBaseSort::sort_column(
    parquet::ParquetFileReader* file_reader,
    shared_ptr<parquet::FileMetaData> metadata, uint32_t col_idx) {
  // Input checking
  if (col_idx >= metadata->num_columns()) {
    std::cerr << "Column index out of range." << std::endl;
    throw std::runtime_error("Column index out of range.");
  }
  // Use multithreading to accumulate chunk sorting results
  auto num_rg = metadata->num_row_groups();
  this->dict_value_counts.resize(num_rg);
  this->index_offset.resize(num_rg);
  vector<future<vector<MergeElement>>> futures(num_rg);
  for (size_t i = 0; i < num_rg; i++) {
    // Launch a thread to do sorting
    futures[i] = std::async(std::launch::async, &CountBaseSort::sort_chunk,
                            this, file_reader->RowGroup(i), col_idx, i);
  }
  // Count the index offset
  int current_offset = 0;
  for (int i = 0; i < num_rg; i++) {
    this->index_offset[i] = current_offset;
    current_offset += metadata->RowGroup(i)->num_rows();
  }
  // Wait for all threads to finish
  vector<vector<MergeElement>> streams;
  streams.reserve(num_rg);
  for (size_t i = 0; i < num_rg; i++) {
    streams.push_back(std::move(futures[i].get()));
  }
  // Merge the results from different streams
  auto merge_result = merge_streams(streams);
  // Construct a Index Mapping based on counting results.
  auto offset_map = get_offset_mapping(std::move(merge_result), num_rg);
  // Re-iterate the column to get the final index list
  int num_rows = metadata->num_rows();
  size_t current_index = 0;
  vector<IndexType> result(num_rows);
  vector<std::future<void>> threads(num_rg);
  for (int rg_idx = 0; rg_idx < num_rg; ++rg_idx) {
    IndexType* offset = offset_map[rg_idx];
    // Use multi-threading to update global index
    // threads[rg_idx] = std::async(
    //     std::launch::async, &CountBaseSort::set_global_index_for_chunk, this,
    //     offset, rg_idx, col_idx, std::ref(result),
    //     file_reader->RowGroup(rg_idx)->GetColumnPageReader(col_idx));
    // Get Dictionary Page
    auto row_group = file_reader->RowGroup(rg_idx);
    auto column_pager = row_group->GetColumnPageReader(col_idx);
    // Iterate through pages to place the position into the result
    // Skip first page
    shared_ptr<parquet::Page> page = column_pager->NextPage();
    vector<int32_t> values;  // Temp container for index decoding
    auto index_decoder = parquet::MakeDictDecoder<parquet::Int64Type>();
    while (page = column_pager->NextPage(), page != nullptr) {
      if (page->type() == parquet::PageType::DATA_PAGE) {
        auto index_page = std::static_pointer_cast<parquet::DataPage>(page);
        index_decoder->SetData(static_cast<int>(index_page->num_values()),
                               index_page->data(), index_page->size());
        values.resize(index_page->num_values());
        index_decoder->DecodeIndices(index_page->num_values(), values.data());
        for (auto& dict_idx : values) {
          result[offset[dict_idx]] = current_index;
          offset[dict_idx]++;
          current_index++;
        }
      } else if (page->type() == parquet::PageType::INDEX_PAGE) {
        // Skip Index page for now. Can be used to optimize
        continue;
      } else {
        string page_type = ParquetPageTypeToString(page->type());
        throw std::runtime_error(
            "Unexpected page type. Expecting: IndexPage. Met: " + page_type);
      }
    }  // End of page iteration within Chunk
  }  // End of RowGroup iteration
  //// Wait threads to finish
  // for (auto& thread : threads) {
  //   thread.get();
  // }
  // Free offset_map
  for (int i = 0; i < metadata->num_row_groups(); i++) {
    delete[] offset_map[i];
  }
  delete[] offset_map;
  return result;
}

vector<CountBaseSort::MergeElement> CountBaseSort::merge_streams(
    const vector<vector<MergeElement>>& streams) {
  vector<MergeElement> result;
  if (streams.empty()) {
    return result;
  } else if (streams.size() == 1) {
    return std::move(streams[0]);
  }
  size_t total_size = 0;
  for (const auto& stream : streams) {
    total_size += stream.size();
  }
  result.reserve(total_size);
  using StreamIterator = vector<MergeElement>::const_iterator;
  auto comp = [](const pair<StreamIterator, StreamIterator>& a,
                 const pair<StreamIterator, StreamIterator>& b) {
    return a.first->key > b.first->key;
  };
  // MinHeap for merging, pair: {first, second} of the stream. We use first
  // iterator to iterate through stream elements and use second iterator to
  // indicate the end of the stream.
  std::priority_queue<pair<StreamIterator, StreamIterator>,
                      vector<pair<StreamIterator, StreamIterator>>,
                      decltype(comp)>
      minHeap(comp);
  for (const auto& stream : streams) {
    if (!stream.empty()) {
      minHeap.emplace(stream.begin(), stream.end());
    }
  }
  // minStream.first: Current element in the stream
  // minStream.second: End of the stream
  while (!minHeap.empty()) {
    auto minStream = minHeap.top();
    minHeap.pop();
    result.push_back(*minStream.first);
    ++minStream.first;  // Move to the next element in this stream
    if (minStream.first != minStream.second) {
      minHeap.push(minStream);  // Push back the updated pair if not at end
    }
  }
  return result;
}

IndexType** CountBaseSort::get_offset_mapping(
    vector<CountBaseSort::MergeElement>&& index_list, uint32_t num_rg) {
  // index_list.size() := Num of RowGroups
  IndexType** offset_mapping = new IndexType*[num_rg];
  uint64_t current_offset = 0;
  // Iterate through RowGroups
  for (int i = 0; i < num_rg; i++) {
    offset_mapping[i] = new IndexType[dict_value_counts[i]];
  }
  // Iterate through the index lists
  for (auto& element : index_list) {
    offset_mapping[element.row_group_index][element.dict_index] =
        current_offset;
    current_offset += element.count;
  }
  return offset_mapping;
}

vector<CountBaseSort::MergeElement> CountBaseSort::sort_chunk(
    shared_ptr<parquet::RowGroupReader> row_gp_reader, uint32_t col_idx,
    uint32_t row_group_index) {
  auto column_pager = row_gp_reader->GetColumnPageReader(col_idx);
  auto column_meta = row_gp_reader->metadata()->ColumnChunk(col_idx);
  auto type = column_meta->type();
  shared_ptr<parquet::Page> first_page = column_pager->NextPage();
  if (first_page->type() != parquet::PageType::DICTIONARY_PAGE) {
    throw std::runtime_error("Expecting the first page being dictionary page");
  }
  auto dict_page =
      std::static_pointer_cast<parquet::DictionaryPage>(first_page);
  int32_t num_dict_values = dict_page->num_values();
  this->dict_value_counts[row_group_index] = num_dict_values;
  if (type == parquet::Type::INT64) {
    auto decoder =
        parquet::MakeTypedDecoder<parquet::Int64Type>(parquet::Encoding::PLAIN);
    decoder->SetData(dict_page->num_values(), dict_page->data(),
                     dict_page->size());
    vector<int64_t> dict(num_dict_values);
    decoder->Decode(&dict[0], num_dict_values);
    auto count_future = std::async(std::launch::async, count_elements,
                                   std::move(column_pager), num_dict_values);
    // Local Dict sorting. (Can use SIMD bitonic to accelerate)
    // Here we use result vector to record index first, then store the count
    // from count_values function. By doing so we can save one copy step.
    vector<CountBaseSort::MergeElement> result;
    result.reserve(num_dict_values);
    for (int i = 0; i < num_dict_values; ++i) {
      result.emplace_back(dict[i], 0, row_group_index, i);
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.key < b.key; });

    // Wait for count_values to finish
    auto count_vector = std::move(count_future.get());
    // Get count values, store them into the result vector.
    // Note that after sorting the result[i].count is the original index for
    // that element.
    for (int i = 0; i < num_dict_values; i++) {
      result[i].count = count_vector[result[i].dict_index];
    }
    return result;
  } else {
    throw std::runtime_error(
        "Met Unsupported column type during page iteration.");
  }
}

vector<uint64_t> CountBaseSort::count_elements(
    unique_ptr<parquet::PageReader> pager, const size_t dict_size) {
  vector<uint64_t> result(dict_size, 0);
  shared_ptr<parquet::Page> page;  // Used for page iteration
  vector<int32_t> values;          // Temp container for index decoding
  // Decoder used to decode index page
  auto index_decoder = parquet::MakeDictDecoder<parquet::Int64Type>();
  while (page = pager->NextPage(), page != nullptr) {
    if (page->type() == parquet::PageType::DATA_PAGE) {
      auto index_page = std::static_pointer_cast<parquet::DataPage>(page);
      index_decoder->SetData(static_cast<int>(index_page->num_values()),
                             index_page->data(), index_page->size());
      values.resize(index_page->num_values());
      index_decoder->DecodeIndices(index_page->num_values(), values.data());
      for (auto& value : values) {
        result[value]++;
      }
    } else if (page->type() == parquet::PageType::INDEX_PAGE) {
      // Skip Index page for now. Can be used to optimize
      continue;
    } else {
      string page_type = ParquetPageTypeToString(page->type());
      throw std::runtime_error(
          "Unexpected page type. Expecting: IndexPage. Met: " + page_type);
    }
  }
  return result;
}

vector<IndexType> IndexBaseSort::sort_column(
    parquet::ParquetFileReader* file_reader,
    shared_ptr<parquet::FileMetaData> metadata, uint32_t col_idx) {
  // Input checking
  if (col_idx >= metadata->num_columns()) {
    std::cerr << "Column index out of range." << std::endl;
    throw std::runtime_error("Column index out of range.");
  }
  // Use multithreading to accumulate chunk sorting results
  auto num_rg = metadata->num_row_groups();
  vector<future<vector<MergeElement>>> futures(num_rg);
  for (size_t i = 0; i < num_rg; i++) {
    // Launch a thread to do sorting
    futures[i] = std::async(std::launch::async, IndexBaseSort::sort_chunk,
                            file_reader->RowGroup(i), col_idx, index_offset[i]);
  }
  // Wait for all threads to finish
  vector<vector<MergeElement>> results;
  results.reserve(num_rg);
  for (size_t i = 0; i < num_rg; i++) {
    results.push_back(std::move(futures[i].get()));
  }
  // Merge the results
  return IndexBaseSort::merge_streams(results, metadata->num_rows());
}

vector<IndexType> IndexBaseSort::merge_streams(
    const vector<vector<MergeElement>>& streams, const size_t res_size) {
  vector<IndexType> result;
  result.reserve(res_size);  // Avoid frequent reinitialization
  if (streams.empty()) {
    return result;
  } else if (streams.size() == 1) {
    for (auto& element : streams[0]) {
      result.insert(result.end(), element.payload->begin(),
                    element.payload->end());
    }
    return result;
  }
  // Pair: {MergeElement, stream_index}, need to use stream_index to track
  // current stage
  auto comp = [](const pair<MergeElement, size_t>& a,
                 const pair<MergeElement, size_t>& b) {
    return a.first.key > b.first.key;
  };
  std::priority_queue<pair<MergeElement, size_t>,
                      vector<pair<MergeElement, size_t>>, decltype(comp)>
      minHeap(comp);

  // Vector to keep track of current index in each stream
  vector<size_t> streamIndices(streams.size(), 0);

  // Initialize the heap with the first element from each stream
  for (size_t i = 0; i < streams.size(); ++i) {
    if (!streams[i].empty()) {
      minHeap.push({streams[i][0], i});
    }
  }

  // Process elements until the heap is empty or reached res_size
  while (!minHeap.empty() && result.size() < res_size) {
    auto top = minHeap.top();
    minHeap.pop();

    MergeElement currentElement = top.first;
    size_t streamIndex = top.second;

    // Add all indices from the payload to the result
    result.insert(result.end(), currentElement.payload->begin(),
                  currentElement.payload->end());

    streamIndices[streamIndex]++;
    if (streamIndices[streamIndex] < streams[streamIndex].size()) {
      minHeap.push(
          {streams[streamIndex][streamIndices[streamIndex]], streamIndex});
    }
  }
  return result;
}

vector<deque<IndexType>*> IndexBaseSort::get_index_lists(
    unique_ptr<parquet::PageReader> pager, const size_t dict_size,
    const uint32_t index_offset) {
  vector<deque<IndexType>*> result(dict_size);
  for (auto& item : result) {
    item = new deque<IndexType>();
  }
  shared_ptr<parquet::Page> page;  // Used for page iteration
  vector<int32_t> values;          // Temp container for index values
  values.reserve(10000);
  uint32_t current_index = index_offset;  // Current index within the chunk
  // Decoder used to decode index page
  auto index_decoder = parquet::MakeDictDecoder<parquet::Int64Type>();
  while (page = pager->NextPage(), page != nullptr) {
    if (page->type() == parquet::PageType::DATA_PAGE) {
      auto index_page = std::static_pointer_cast<parquet::DataPage>(page);
      index_decoder->SetData(static_cast<int>(index_page->num_values()),
                             index_page->data(), index_page->size());
      values.resize(index_page->num_values());
      index_decoder->DecodeIndices(index_page->num_values(), values.data());
      for (auto& value : values) {
        result[value]->push_back(current_index);
        current_index++;
      }
    } else if (page->type() == parquet::PageType::INDEX_PAGE) {
      // Skip Index page for now. Can be used to optimize
      continue;
    } else {
      string page_type = ParquetPageTypeToString(page->type());
      throw std::runtime_error(
          "Unexpected page type. Expecting: IndexPage. Met: " + page_type);
    }
  }
  return result;
}

vector<IndexBaseSort::MergeElement> IndexBaseSort::sort_chunk(
    shared_ptr<parquet::RowGroupReader> row_gp_reader, uint32_t col_idx,
    int index_offset) {
  auto column_pager = row_gp_reader->GetColumnPageReader(col_idx);
  auto column_meta = row_gp_reader->metadata()->ColumnChunk(col_idx);
  auto type = column_meta->type();
  shared_ptr<parquet::Page> first_page = column_pager->NextPage();
  if (first_page->type() != parquet::PageType::DICTIONARY_PAGE) {
    throw std::runtime_error("Expecting the first page being dictionary page");
  }
  auto dict_page =
      std::static_pointer_cast<parquet::DictionaryPage>(first_page);
  int32_t dict_count = dict_page->num_values();
  if (type == parquet::Type::INT64) {
    auto decoder =
        parquet::MakeTypedDecoder<parquet::Int64Type>(parquet::Encoding::PLAIN);
    decoder->SetData(dict_count, dict_page->data(), dict_page->size());
    vector<int64_t> dict(dict_count);
    decoder->Decode(&dict[0], dict_count);
    auto count_future =
        std::async(std::launch::async, IndexBaseSort::get_index_lists,
                   std::move(column_pager), dict_count, index_offset);
    // Local Dict sorting.
    // pair.first is the dictionary key
    // pair.second is the original index
    vector<pair<int64_t, int>> pairs(dict_count);
    for (int i = 0; i < dict_count; ++i) {
      pairs[i].first = dict[i];
      pairs[i].second = i;
    }

    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Wait for count_values to finish
    auto index_count = std::move(count_future.get());
    // Get result
    vector<MergeElement> result;
    result.reserve(dict_count);
    for (int i = 0; i < dict_count; i++) {
      result.emplace_back(pairs[i].first, index_count[pairs[i].second]);
    }

    return result;
  } else {
    throw std::runtime_error(
        "Met Unsupported column type during page iteration.");
  }
}

vector<uint32_t> ParquetSorter::sort_column(uint32_t col_idx) {
  // Input checking
  if (col_idx >= src_metadata->num_columns()) {
    std::cerr << "Column index out of range." << std::endl;
    throw std::runtime_error("Column index out of range.");
  }
  return sorter->sort_column(file_reader.get(), src_metadata, col_idx);
}

ParquetSorter::ParquetSorter()
    : output_file(""), src_metadata(nullptr), file_reader(nullptr) {}

shared_ptr<ParquetSorter> ParquetSorter::create(
    const string& input_file, const string& output_file,
    const SortStrategy::SortType& sort_type) {
  shared_ptr<ParquetSorter> sorter(new ParquetSorter());
  sorter->output_file = output_file;
  sorter->input_file = input_file;
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  auto state = arrow::io::ReadableFile::Open(input_file);
  if (!state.ok()) {
    std::cerr << "Failed to open input file." << std::endl;
    throw std::runtime_error("Failed to open input parquet file");
  }
  file = state.ValueOrDie();
  std::unique_ptr<parquet::ParquetFileReader> reader =
      parquet::ParquetFileReader::Open(file);
  sorter->src_metadata = reader->metadata();
  sorter->file_reader = std::move(reader);

  if (sort_type == SortStrategy::SortType::COUNT_BASE) {
    sorter->set_strategy(std::make_unique<CountBaseSort>());
  } else if (sort_type == SortStrategy::SortType::INDEX_BASE) {
    int num_row_groups = sorter->src_metadata->num_row_groups();
    vector<IndexType> index_offsets(num_row_groups, 0);
    for (int i = 1; i < num_row_groups; i++) {
      index_offsets[i] = index_offsets[i - 1] +
                         sorter->src_metadata->RowGroup(i - 1)->num_rows();
    }
    sorter->set_strategy(
        std::make_unique<IndexBaseSort>(std::move(index_offsets)));
  } else {
    throw std::runtime_error("Unsupported sort type.");
  }
  return sorter;
}

arrow::Status ParquetSorter::write(vector<uint32_t>&& index_list) {
  // Read the entire input file into an Arrow table
  std::shared_ptr<arrow::io::RandomAccessFile> infile;
  ARROW_ASSIGN_OR_RAISE(infile,
                        arrow::io::ReadableFile::Open(this->input_file));
  unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  // PARQUET_THROW_NOT_OK(parquet::arrow::FileReader::Make(
  //     arrow::default_memory_pool(), std::move(this->file_reader), &reader));

  shared_ptr<arrow::Table> parquet_table;
  // Read the table.
  PARQUET_THROW_NOT_OK(reader->ReadTable(&parquet_table));
  // Create an Arrow array from the index_list
  auto index_size = index_list.size();
  shared_ptr<arrow::Buffer> index_buffer =
      arrow::Buffer::FromVector(std::move(index_list));
  auto index_array =
      std::make_shared<arrow::UInt32Array>(index_size, index_buffer);

  // Transform the table using the index
  arrow::compute::TakeOptions options;
  arrow::compute::ExecContext context(arrow::default_memory_pool());
  arrow::Datum result;
  ARROW_ASSIGN_OR_RAISE(result, arrow::compute::Take(parquet_table, index_array,
                                                     options, &context));
  shared_ptr<arrow::Table> sorted_table = result.table();

  // Write to Parquet Table
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  ARROW_ASSIGN_OR_RAISE(outfile,
                        arrow::io::FileOutputStream::Open(this->output_file));
  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
      *sorted_table.get(), arrow::default_memory_pool(), outfile));

  // Close the writer
  PARQUET_THROW_NOT_OK(outfile->Close());
  return arrow::Status::OK();
}
/**
 * @deprecated Old implementation of vector based index sorting.
 * @brief: This function will iterate a ColumnChunk and record the index of each
 * elements of the dictionary. For example, if dictionary is ['A','B'] and data
 * are [0,1,1,0,0] then the result will be [[0,3,4],[1,2]] where the first
 * vector is the index of 'A' and the second vector is the index of 'B'. This
 * result can be used to generate SortIndex in the final stage.
 * @param pager: PageReader for the column chunk, assumes the dictionary page
 * has been consumed
 * @param dict_size: The size of the dictionary
 * @param index_offset: The offset of the index in the global layout.
 */
// template <typename ParquetType>
// vector<vector<uint32_t>> _count_index(unique_ptr<parquet::PageReader> pager,
//                                       const size_t dict_size,
//                                       const uint32_t index_offset) {
//   vector<vector<uint32_t>> index(dict_size);
//   std::shared_ptr<parquet::Page> page;  // Used for page iteration
//   vector<int32_t> values;               // Temp container for index values
//   values.reserve(10000);
//   uint32_t current_index = 0;  // Current index within the chunk
//   // Decoder used to decode index page
//   auto index_decoder = parquet::MakeDictDecoder<ParquetType>();
//   while (page = pager->NextPage(), page != nullptr) {
//     if (page->type() == parquet::PageType::DATA_PAGE) {
//       auto index_page = std::static_pointer_cast<parquet::DataPage>(page);
//       index_decoder->SetData(static_cast<int>(index_page->num_values()),
//                              index_page->data(), index_page->size());
//       values.resize(index_page->num_values());
//       index_decoder->DecodeIndices(index_page->num_values(), values.data());
//       for (auto& value : values) {
//         index[value].push_back(current_index);
//         current_index++;
//       }
//     } else if (page->type() == parquet::PageType::INDEX_PAGE) {
//       // Skip Index page for now. Can be used to optimize
//       continue;
//     } else {
//       string page_type = ParquetPageTypeToString(page->type());
//       throw std::runtime_error(
//           "Unexpected page type. Expecting: IndexPage. Met: " + page_type);
//     }
//   }
//   return std::move(index);
// }
}  // namespace whippet_sort
