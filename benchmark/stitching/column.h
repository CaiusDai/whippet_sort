
#ifndef STITCH_COLUMN_H
#define STITCH_COLUMN_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using std::memcmp;
using std::vector;
typedef vector<uint32_t> RawColumn;
/**
 * @file column.h
 * @brief Define the column class and relating helper functions for experiments
 *
 */
namespace stitch {

struct SortingGroup {
  size_t start_idx;
  size_t length;
};

struct SortingState {
  vector<SortingGroup> groups;
  vector<uint32_t> indices;  // permutation from first phase
};

template <size_t WIDTH>
struct Tuple {
  uint32_t rowID;
  uint32_t values[WIDTH];
};

typedef Tuple<1> Tuple1;
typedef Tuple<2> Tuple2;
typedef Tuple<3> Tuple3;
typedef Tuple<4> Tuple4;

class Column {
 public:
  Column()
      : data(nullptr), num_values(0), compare_factor(1), record_groups(false) {}
  ~Column() {
    if (data != nullptr) {
      delete[] data;
    }
  }

  Column(const Column& other)
      : num_values(other.num_values),
        compare_factor(other.compare_factor),
        record_groups(other.record_groups) {
    if (other.data != nullptr) {
      const size_t total_size = num_values * (compare_factor + 1);
      data = new uint32_t[total_size];
      std::memcpy(data, other.data, total_size * sizeof(uint32_t));
    } else {
      data = nullptr;
    }
  }

  Column& operator=(const Column& other) {
    if (this != &other) {
      if (data != nullptr) delete[] data;
      num_values = other.num_values;
      compare_factor = other.compare_factor;
      record_groups = other.record_groups;
      if (other.data != nullptr) {
        const size_t total_size = num_values * (compare_factor + 1);
        data = new uint32_t[total_size];
        std::memcpy(data, other.data, total_size * sizeof(uint32_t));
      } else {
        data = nullptr;
      }
    }
    return *this;
  }

  Column(Column&& other) noexcept
      : data(other.data),
        num_values(other.num_values),
        compare_factor(other.compare_factor),
        record_groups(other.record_groups) {
    other.data = nullptr;
    other.num_values = 0;
    other.compare_factor = 1;
  }

  Column& operator=(Column&& other) noexcept {
    if (this != &other) {  // Self-assignment check
      if (data != nullptr) delete[] data;
      data = other.data;
      num_values = other.num_values;
      compare_factor = other.compare_factor;
      record_groups = other.record_groups;
      other.data = nullptr;
      other.num_values = 0;
      other.compare_factor = 1;
    }
    return *this;
  }

  static Column stitch(const vector<RawColumn*>& cols,
                       const vector<uint32_t>& indices) {
    Column result;
    if (cols.empty() || indices.empty()) return result;
    // RawColumn only consists of 1 element each
    result.compare_factor = cols.size();
    result.num_values = indices.size();

    // Memory allocation
    const size_t tuple_size = result.compare_factor + 1;
    result.data = new uint32_t[result.num_values * tuple_size];

    // Stitching
    uint32_t* curr_tuple = result.data;
    for (size_t i = 0; i < indices.size(); i++) {
      curr_tuple[0] = indices[i];
      for (size_t col = 0; col < cols.size(); col++) {
        curr_tuple[col + 1] = (*cols[col])[indices[i]];
      }

      curr_tuple += tuple_size;
    }

    return result;
  }

  vector<uint32_t> get_index_only() const {
    vector<uint32_t> result;
    result.resize(num_values);
    uint32_t* curr_tuple = data;
    const uint32_t offset = compare_factor + 1;
    for (size_t i = 0; i < num_values; i++) {
      result[i] = curr_tuple[0];
      curr_tuple += offset;
    }
    return result;
  }

  SortingState get_groups_and_index() const {
    SortingState state;
    state.indices.resize(num_values);
    size_t start = 0;
    uint32_t* curr_tuple = data;
    const uint32_t tuple_offset = compare_factor + 1;
    for (size_t i = 0; i < num_values - 1; i++) {
      // Update index
      state.indices[i] = curr_tuple[0];
      // Compare current tuple with next tuple
      if (memcmp(curr_tuple + 1, curr_tuple + tuple_offset + 1,
                 compare_factor * sizeof(uint32_t)) != 0) {
        state.groups.push_back({start, i - start + 1});
        start = i + 1;
      }
      curr_tuple += tuple_offset;
    }
    state.indices[num_values - 1] = curr_tuple[0];
    state.groups.push_back({start, num_values - start});
    return state;
  }

  SortingState get_groups_and_index(vector<SortingGroup>& group) const {
    SortingState state;
    state.indices.resize(num_values);
    size_t start = 0;
    uint32_t* curr_tuple = data;
    const uint32_t tuple_offset = compare_factor + 1;
    for (size_t group_idx = 0; group_idx < group.size(); group_idx++) {
      if (group[group_idx].length == 1) {
        state.indices[start] = curr_tuple[0];
        state.groups.push_back({start, 1});
        start++;
        curr_tuple += tuple_offset;
        continue;
      } else {
        for (size_t base = start;
             base < group[group_idx].start_idx + group[group_idx].length;
             base++) {
          state.indices[base] = curr_tuple[0];
          if (memcmp(curr_tuple + 1, curr_tuple + tuple_offset + 1,
                     compare_factor * sizeof(uint32_t)) != 0) {
            state.groups.push_back({start, base - start + 1});
            start = base + 1;
          }
          curr_tuple += tuple_offset;
          if (base == start + group[group_idx].length - 1) {
            state.groups.push_back({start, base - start + 1});
            state.indices[base] = curr_tuple[0];
          }
        }
      }
    }
    return state;
  }

  void sort(const std::vector<SortingGroup>& groups) {
    const uint32_t tuple_size = compare_factor + 1;

    auto sort_groups = [&]<size_t W>(Tuple<W>* /*dummy*/) {
      for (const auto& group : groups) {
        if (group.length == 1) continue;

        auto* tuples =
            reinterpret_cast<Tuple<W>*>(data + group.start_idx * tuple_size);
        std::sort(tuples, tuples + group.length,
                  [this](const Tuple<W>& a, const Tuple<W>& b) {
                    return memcmp(&a.values[0], &b.values[0],
                                  compare_factor * sizeof(uint32_t)) < 0;
                  });
      }
    };

    switch (compare_factor) {
      case 1:
        sort_groups(static_cast<Tuple1*>(nullptr));
        break;
      case 2:
        sort_groups(static_cast<Tuple2*>(nullptr));
        break;
      case 3:
        sort_groups(static_cast<Tuple3*>(nullptr));
        break;
      case 4:
        sort_groups(static_cast<Tuple4*>(nullptr));
        break;
      default:
        break;
    }
  }

  void sort() {
    auto sort_helper = [this]<size_t W>(Tuple<W>* tuples) {
      std::sort(tuples, tuples + num_values,
                [](const Tuple<W>& a, const Tuple<W>& b) {
                  return memcmp(a.values, b.values, W * sizeof(uint32_t)) < 0;
                });
    };

    switch (compare_factor) {
      case 1:
        sort_helper(reinterpret_cast<Tuple1*>(data));
        break;
      case 2:
        sort_helper(reinterpret_cast<Tuple2*>(data));
        break;
      case 3:
        sort_helper(reinterpret_cast<Tuple3*>(data));
        break;
      case 4:
        sort_helper(reinterpret_cast<Tuple4*>(data));
        break;
    }
  }

  void print_data() {
    const uint32_t tuple_size = compare_factor + 1;
    uint32_t* curr_tuple = data;
    for (size_t i = 0; i < num_values; i++) {
      std::cout << "[" << curr_tuple[0] << "] ";
      for (size_t j = 1; j < tuple_size; j++) {
        std::cout << curr_tuple[j] << " ";
      }
      std::cout << std::endl;
      curr_tuple += tuple_size;
    }
  }

  // fields:
  uint32_t* data;
  bool record_groups = false;
  size_t num_values;
  size_t compare_factor;  // Draft use only
};

}  // namespace stitch

#endif  // STITCH_COLUMN_H