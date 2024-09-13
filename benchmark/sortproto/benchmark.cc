#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <unistd.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "parquet/column_page.h"
#include "parquet/column_reader.h"
#include "parquet/encoding.h"
#include "parquet/file_reader.h"
#include "parquet/types.h"
#include "parquet_sorter.h"
using std::pair;

const string _WHIPPET_COUNT_OUT = "whippet_out_count.parquet";
const string _WHIPPET_INDEX_OUT = "whippet_out_index.parquet";
const string _ARROW_OUT = "arrow_out.parquet";
/**
 * @brief Check the validation of an index list.
 * By checking if it has correct value range (0 to n-1) and no
 * duplicated values.
 */
bool is_valid_index_list(
    const whippet_sort::IndexType max,
    const std::vector<whippet_sort::IndexType>& index_list) {
  if (index_list.size() > max) {
    std::cerr << "Index list size exceeds the maximum index: "
              << index_list.size() << std::endl;
    return false;
  }
  std::vector<bool> index_check(max, false);
  for (auto& index : index_list) {
    if (index < 0 || index >= max) {
      std::cerr << "[Error] Invalid index found: " << index << std::endl;
      return false;
    } else if (index_check[index]) {
      std::cerr << "[Error] Duplicate index found: " << index << std::endl;
      return false;
    }
    index_check[index] = true;
  }
  return true;
}

void drop_file_cache(const std::string& file_path) {
  std::string command =
      "dd of=" + file_path +
      " oflag=nocache conv=notrunc,fdatasync status=none count=0";
  auto drop_cache = system(command.c_str());
  if (drop_cache != 0) {
    std::cerr << "Failed to drop file cache. Error code: " << drop_cache
              << std::endl;
  }
}

arrow::Status arrow_sorting(const std::string& input_file,
                            const std::string& output_file) {
  // Open the input file
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(input_file));

  // Create a ParquetFileReader
  std::unique_ptr<parquet::arrow::FileReader> reader;
  ARROW_RETURN_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

  // Read the entire file as a Table
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));

  int sort_column_index = 0;

  // Get the column to sort
  std::shared_ptr<arrow::ChunkedArray> column =
      table->column(sort_column_index);
  // Sort the column
  arrow::compute::ExecContext ctx;
  arrow::compute::SortOptions sort_options;
  arrow::compute::TakeOptions take_options;
  ARROW_ASSIGN_OR_RAISE(auto sort_indices, arrow::compute::SortIndices(
                                               column, sort_options, &ctx));
  ARROW_ASSIGN_OR_RAISE(auto result, arrow::compute::Take(table, sort_indices,
                                                          take_options, &ctx));

  shared_ptr<arrow::Table> sorted_table = result.table();

  ARROW_ASSIGN_OR_RAISE(auto outfile,
                        arrow::io::FileOutputStream::Open(output_file));
  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
      *sorted_table, arrow::default_memory_pool(), outfile));
  // Close the writer
  PARQUET_THROW_NOT_OK(outfile->Close());
  return arrow::Status::OK();
}

void whippet_sorting(const std::string& input_file,
                     const std::string& output_file,
                     whippet_sort::SortStrategy::SortType sort_type) {
  using namespace whippet_sort;
  auto sorter = ParquetSorter::create(input_file, output_file, sort_type);
  auto index_list = sorter->sort_column(0);
  auto status = sorter->write(std::move(index_list));
  if (!status.ok()) {
    std::cerr << "Failed to write sorted table to output file." << std::endl;
    throw std::runtime_error("Failed to write sorted table to output file.");
  }
}

template <typename Func>
pair<double, double> benchmark(Func&& func, int num_runs) {
  std::vector<double> durations;
  durations.reserve(num_runs);

  for (int i = 0; i < num_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    double duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    durations.push_back(duration);
  }

  // Calculate average
  double average =
      std::accumulate(durations.begin(), durations.end(), 0.0) / num_runs;

  // Calculate median
  std::sort(durations.begin(), durations.end());
  double median = durations[num_runs / 2];

  return {median, average};
}

/**
 * @brief Given a parquet file, check if a column is sorted or not.
 */
bool is_sorted_column(const std::string& parquet_file,
                      size_t sorted_column_index) {
  auto state = arrow::io::ReadableFile::Open(parquet_file);
  if (!state.ok()) {
    throw std::runtime_error("Failed to open input parquet file");
  }
  std::shared_ptr<arrow::io::RandomAccessFile> input_file = state.ValueOrDie();

  auto parquet_reader = parquet::ParquetFileReader::Open(input_file);
  auto file_metadata = parquet_reader->metadata();
  if (sorted_column_index >= file_metadata->num_columns()) {
    std::cerr << "Invalid column index." << std::endl;
    return false;
  }

  // Read the column data
  const size_t batch_size{1000};
  int64_t values[batch_size];
  int64_t values_read = 0;
  int64_t previous_value = std::numeric_limits<int64_t>::min();
  for (size_t i = 0; i < file_metadata->num_row_groups(); i++) {
    std::shared_ptr<parquet::ColumnReader> column_reader =
        parquet_reader->RowGroup(i)->Column(sorted_column_index);
    parquet::Int64Reader* int64_reader =
        static_cast<parquet::Int64Reader*>(column_reader.get());

    while (int64_reader->HasNext()) {
      int64_reader->ReadBatch(batch_size, nullptr, nullptr, &values[0],
                              &values_read);
      for (int64_t i = 0; i < values_read; ++i) {
        if (values[i] < previous_value) {
          std::cerr << "Column is not sorted at index " << i << std::endl;
          return false;
        }
        previous_value = values[i];
      }
    }
  }
  return true;
}

bool check_encoding_consistency(const string& input_file,
                                parquet::Encoding::type encoding,
                                const size_t col_index) {
  // Open the input file
  PARQUET_ASSIGN_OR_THROW(auto infile,
                          arrow::io::ReadableFile::Open(input_file));

  // Create a ParquetFileReader
  std::unique_ptr<parquet::ParquetFileReader> reader =
      parquet::ParquetFileReader::Open(infile);
  auto meta = reader->metadata();
  std::cout << "There are " << meta->num_row_groups() << " row groups"
            << std::endl;
  uint64_t num_page_processed = 0;
  // Iterate through row groups and check each column's encoding
  for (int i = 0; i < meta->num_row_groups(); i++) {
    auto row_group_reader = reader->RowGroup(i);
    auto page_reader = row_group_reader->GetColumnPageReader(col_index);
    std::shared_ptr<parquet::Page> page;
    while (page = page_reader->NextPage(), page != nullptr) {
      if (page->type() == parquet::PageType::DATA_PAGE ||
          page->type() == parquet::PageType::DATA_PAGE_V2) {
        auto data_page = std::static_pointer_cast<parquet::DataPage>(page);
        if (data_page->encoding() != encoding) {
          std::cerr << "Encoding mismatch at row group " << i << std::endl;
          return false;
        }
      }
      num_page_processed++;
    }
  }
  std::cout << "Processed in total " << num_page_processed << " pages"
            << std::endl;
  return true;
}

void report_meta(const string& input_file) {
  // Open the input file
  PARQUET_ASSIGN_OR_THROW(auto infile,
                          arrow::io::ReadableFile::Open(input_file));

  // Create a ParquetFileReader
  std::unique_ptr<parquet::ParquetFileReader> reader =
      parquet::ParquetFileReader::Open(infile);
  auto meta = reader->metadata();
  std::cout << "Summarizing metadata for file " << input_file << std::endl;
  std::cout << "There are " << meta->num_columns() << " columns" << std::endl;
  std::cout << "There are " << meta->num_row_groups() << " row groups"
            << std::endl;
  auto schema = meta->schema();
  // Report encodings for each column
  for (int i = 0; i < meta->num_columns(); i++) {
    auto column_type_str =
        parquet::TypeToString(schema->Column(i)->physical_type());
    std::cout << "Column " << i << " has type " << column_type_str << std::endl;
    std::unordered_set<parquet::Encoding::type> encodings;
    for (int j = 0; j < meta->num_row_groups(); j++) {
      auto row_group_reader = reader->RowGroup(j);
      auto page_reader = row_group_reader->GetColumnPageReader(i);
      std::shared_ptr<parquet::Page> page;
      while (page = page_reader->NextPage(), page != nullptr) {
        if (page->type() == parquet::PageType::DATA_PAGE ||
            page->type() == parquet::PageType::DATA_PAGE_V2) {
          auto data_page = std::static_pointer_cast<parquet::DataPage>(page);
          encodings.insert(data_page->encoding());
        }
      }
    }
    std::cout << "Column " << i << " has encodings: \n";
    for (auto& encoding : encodings) {
      auto encoding_str = parquet::EncodingToString(encoding);
      std::cout << " - " << encoding_str << " \n";
    }
  }
}

int main(const int argc, const char* argv[]) {
  nice(-20);
  auto input_file = std::string(argv[1]);
  int num_runs = std::stoi(argv[2]);
  // Report the number of row groups and number of Rows
  // auto sorter = whippet_sort::ParquetSorter::create(
  //     input_file, "output_file",
  //     whippet_sort::SortStrategy::SortType::COUNT_BASE);
  // std::cout << "Number of RowGroups: "
  //           << sorter->file_reader->metadata()->num_row_groups() <<
  //           std::endl;
  // std::cout << "Number of Rows: " <<
  // sorter->file_reader->metadata()->num_rows()
  //           << std::endl;

  // Benchmark Arrow sorting
  // auto [arrow_median, arrow_average] = benchmark(
  //     [&]() {
  //       drop_file_cache(input_file);
  //       PARQUET_THROW_NOT_OK(arrow_sorting(input_file, _ARROW_OUT));
  //     },
  //     num_runs);

  // std::cout << "Arrow sorting - Median: " << arrow_median
  //           << "ms, Average: " << arrow_average << "ms" << std::endl;

  // Benchmark Whippet sorting(CountBaseSort)
  auto [whippet_count_median, whippet_count_average] = benchmark(
      [&]() {
        drop_file_cache(input_file);
        whippet_sorting(input_file, _WHIPPET_COUNT_OUT,
                        whippet_sort::SortStrategy::SortType::COUNT_BASE);
      },
      num_runs);

  std::cout << "Whippet sorting (CountBaseSort) - Median: "
            << whippet_count_median << "ms, Average: " << whippet_count_average
            << "ms" << std::endl;

  // Benchmark Whippet sorting (IndexBaseSort)
  // auto [whippet_index_median, whippet_index_average] = benchmark(
  //     [&]() {
  //       drop_file_cache(input_file);
  //       whippet_sorting(input_file, _WHIPPET_INDEX_OUT,
  //                       whippet_sort::SortStrategy::SortType::INDEX_BASE);
  //     },
  //     num_runs);

  // std::cout << "Whippet sorting (IndexBaseSort) - Median: "
  //           << whippet_index_median << "ms, Average: " <<
  //           whippet_index_average
  //           << "ms" << std::endl;

  // Check correctness
  // bool count_correct = is_sorted_column(_WHIPPET_COUNT_OUT, 0);
  // std::cout << "Count Base Whippet sort correctness: "
  //           << (count_correct ? "Correct" : "Incorrect") << std::endl;
  // bool index_correct = is_sorted_column(_WHIPPET_INDEX_OUT, 0);
  // std::cout << "Index Base Whippet sort correctness: "
  //           << (index_correct ? "Correct" : "Incorrect") << std::endl;
  return 0;
}