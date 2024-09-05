#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using std::deque;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
namespace whippet_sort {
/**
 * @brief Define the type of index list. This is used to store the positions of
 * data for permutation use.
 */
typedef uint32_t IndexType;

/**
 * @brief This is the base class for different sorting strategies.
 * Currently only support sorting by one column. Produce a list of index
 * for permutation use.
 *
 * Implemented Strategies:
 * 1. Counting based sort
 * 2. Index based sort
 */
class SortStrategy {
 public:
  enum class SortType { COUNT_BASE, INDEX_BASE };
  virtual std::vector<IndexType> sort_column(
      parquet::ParquetFileReader* file_reader,
      shared_ptr<parquet::FileMetaData> metadata, uint32_t col_idx) = 0;
};

class CountBaseSort : public SortStrategy {
 public:
  class MergeElement {
   public:
    MergeElement(int64_t key, uint64_t count) : key(key), count(count) {};
    int64_t key;
    uint64_t count;
  };

  vector<IndexType> sort_column(parquet::ParquetFileReader* file_reader,
                                shared_ptr<parquet::FileMetaData> metadata,
                                uint32_t col_idx) override;

  // Following private static functions are declared here for logical grouping
  // and readability
 private:
  /**
   * @brief Sort a column chunk and return the sorted MergeElement list.
   * Result is then be used for merging.
   */
  static vector<MergeElement> sort_chunk(
      shared_ptr<parquet::RowGroupReader> row_gp_reader, uint32_t col_idx);
  /**
   * @brief Count the appearance of each element in the dictionary within a
   * chunk. Currently only supports INT64 type.
   */
  static vector<uint64_t> count_elements(unique_ptr<parquet::PageReader> pager,
                                         const size_t dict_size);
  /**
   * @brief Merge multiple streams of sorted MergeElement list into one.
   */
  static vector<MergeElement> merge_streams(
      const vector<vector<MergeElement>>& streams);

  /**
   * @brief: Helper function of CountBaseSort::sort_column. This function will
   * transfer a list of sorted <key,count> pairs into a key -> offset mapping.
   * This function will be used after merging process. Having this mapping we
   * can then produce a global index list by reiterate the sorting column.
   */
  static unordered_map<int64_t, IndexType> get_offset_mapping(
      vector<CountBaseSort::MergeElement>&& index_list);
};

class IndexBaseSort : public SortStrategy {
 public:
  /**
   * @brief This struct is the output from different ChunkSorter.
   * Each ChunkSorter will produce an array of MergeElement for
   * later merging stage.
   * Members:
   * key: element from the dictionary
   * payload: the pointer to the index list of that dict element. The index is
   * global position index.
   */
  class MergeElement {
   public:
    MergeElement(int64_t key, deque<IndexType>* payload)
        : key(key), payload(payload) {};
    int64_t key;
    deque<IndexType>* payload;
  };
  IndexBaseSort(vector<IndexType>&& index_offset)
      : index_offset(std::move(index_offset)) {}
  IndexBaseSort() = default;
  void set_index_offset(vector<IndexType>&& index_offset) {
    this->index_offset = std::move(index_offset);
  }
  void set_index_offset(const vector<IndexType>& index_offset) {
    this->index_offset = index_offset;
  }

  vector<IndexType> sort_column(parquet::ParquetFileReader* file_reader,
                                shared_ptr<parquet::FileMetaData> metadata,
                                uint32_t col_idx) override;
  // Following private static functions are declared here for logical grouping
  // and readability
 private:
  /**
   * @brief Sort a column chunk and return the sorted MergeElement list for
   * merging.
   */
  static vector<MergeElement> sort_chunk(
      shared_ptr<parquet::RowGroupReader> row_gp_reader, uint32_t col_idx,
      int index_offset);

  /**
   * @brief: This function will iterate a ColumnChunk and record the index of
   * each elements of the dictionary. For example, if dictionary is ['A','B']
   * and data are [0,1,1,0,0] then the result will be [[0,3,4],[1,2]] where the
   * first vector is the index of 'A' and the second vector is the index of 'B'.
   * This result can be used to generate SortIndex in the final stage. Note that
   * the index is the global index in the column. If the chunk resides in the
   * 2nd RowGroup and first RowGroup contains n elements, then the index of the
   * first element in the chunk is n. This function is used in sort_chunk
   * method.
   * @param pager: PageReader for the column chunk, assumes the dictionary page
   * has been consumed.
   * @param dict_size: The size of the dictionary
   * @param index_offset: The offset of the index in the global layout.
   */
  static vector<deque<IndexType>*> get_index_lists(
      unique_ptr<parquet::PageReader> pager, const size_t dict_size,
      const uint32_t index_offset);

  static vector<IndexType> merge_streams(
      const vector<vector<MergeElement>>& streams, const size_t res_size);

  // Private fields:
 private:
  vector<IndexType> index_offset;
};

class ParquetSorter {
 public:
  static shared_ptr<ParquetSorter> create(
      const string& input_file, const string& output_file,
      const SortStrategy::SortType& sort_type);
  shared_ptr<parquet::FileMetaData> get_metadata() const {
    return src_metadata;
  }

  void set_strategy(unique_ptr<SortStrategy>&& strategy) {
    sorter = std::move(strategy);
  }

  ~ParquetSorter() = default;
  // Disable copy & move constructors for now to avoid unexpected behaviour.
  // Adjust later if needed.
  ParquetSorter(const ParquetSorter&) = delete;
  ParquetSorter& operator=(const ParquetSorter&) = delete;
  // Sort the column with the given index and return the sorted index list.
  vector<IndexType> sort_column(uint32_t col_idx);
  // Write the sorted table to the output file using index list
  arrow::Status write(vector<IndexType>&& index_list);

 public:
  unique_ptr<parquet::ParquetFileReader> file_reader;

 private:
  // Only internal use. For construction, use create() instead.
  ParquetSorter();

  // Fields
  unique_ptr<SortStrategy> sorter;
  string output_file;
  string input_file;
  shared_ptr<parquet::FileMetaData> src_metadata;
};

}  // namespace whippet_sort