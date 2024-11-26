#include "column.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

using stitch::Column;

class ColumnTest : public ::testing::Test {
 protected:
  // Helper function to verify sorting correctness
  bool is_sorted_by_columns(const std::vector<RawColumn*>& cols,
                            const std::vector<uint32_t>& row_indices) {
    if (row_indices.empty()) return true;

    for (size_t i = 0; i < row_indices.size() - 1; i++) {
      // Compare each column value for adjacent rows
      for (const auto* col : cols) {
        uint32_t curr_val = (*col)[row_indices[i]];
        uint32_t next_val = (*col)[row_indices[i + 1]];
        if (curr_val < next_val) break;
        if (curr_val > next_val) {
          return false;
        }
      }
    }
    return true;
  }
};

// Test basic stitching functionality
TEST_F(ColumnTest, SingleRoundBasicStitch) {
  std::vector<uint32_t> col1_data = {1, 2, 3};
  std::vector<uint32_t> col2_data = {4, 5, 6};
  RawColumn raw_col1(col1_data);
  RawColumn raw_col2(col2_data);

  std::vector<RawColumn*> cols = {&raw_col1, &raw_col2};
  std::vector<uint32_t> indices = {0, 1, 2};

  Column stitched = Column::stitch(cols, indices);
  auto state = stitched.get_groups_and_index();

  ASSERT_EQ(state.indices.size(), 3);
  ASSERT_EQ(state.groups.size(), 3);

  for (const auto& group : state.groups) {
    ASSERT_EQ(group.length, 1);
  }
}

// Test sorting with duplicate values
TEST_F(ColumnTest, SingleRoundSortWithDuplicates) {
  std::vector<uint32_t> col1_data = {2, 1, 4, 1, 4, 2};
  std::vector<uint32_t> col2_data = {3, 3, 4, 4, 4, 4};
  RawColumn raw_col1(col1_data);
  RawColumn raw_col2(col2_data);

  std::vector<RawColumn*> cols = {&raw_col1, &raw_col2};
  std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};
  Column stitched = Column::stitch(cols, indices);
  auto state = stitched.get_groups_and_index();
  ASSERT_EQ(state.indices.size(), 6);
  ASSERT_EQ(state.groups.size(), 6);
  // Sort the groups
  stitched.sort();
  // Verify sorting correctness
  state = stitched.get_groups_and_index();
  ASSERT_TRUE(is_sorted_by_columns(cols, state.indices));
  // Verify grouping
  ASSERT_EQ(state.indices.size(), 6);
  ASSERT_EQ(state.groups.size(), 5);
}

TEST_F(ColumnTest, SingleRoundThreeColumnStitching) {
  std::vector<uint32_t> col1_data = {1, 2, 3};
  std::vector<uint32_t> col2_data = {4, 5, 6};
  std::vector<uint32_t> col3_data = {7, 8, 9};
  RawColumn raw_col1(col1_data);
  RawColumn raw_col2(col2_data);
  RawColumn raw_col3(col3_data);

  std::vector<RawColumn*> cols = {&raw_col1, &raw_col2, &raw_col3};
  std::vector<uint32_t> indices = {0, 1, 2};

  Column stitched = Column::stitch(cols, indices);
  auto state = stitched.get_groups_and_index();

  ASSERT_EQ(state.indices.size(), 3);
  ASSERT_EQ(state.groups.size(), 3);

  for (const auto& group : state.groups) {
    ASSERT_EQ(group.length, 1);
  }
}

// Test large random input
TEST_F(ColumnTest, SingleRoundLargeRandomInput) {
  const size_t num_rows = 1000;
  const size_t num_cols = 3;

  // Generate random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 100);

  std::vector<std::vector<uint32_t>> col_data(num_cols);
  std::vector<RawColumn*> raw_cols;
  std::vector<uint32_t> indices(num_rows);

  // Initialize columns with random data
  for (size_t i = 0; i < num_cols; i++) {
    col_data[i].resize(num_rows);
    for (size_t j = 0; j < num_rows; j++) {
      col_data[i][j] = dis(gen);
    }
    raw_cols.push_back(new RawColumn(col_data[i]));
  }

  // Initialize indices
  for (size_t i = 0; i < num_rows; i++) {
    indices[i] = i;
  }

  // Create and sort stitched column
  Column stitched = Column::stitch(raw_cols, indices);
  auto state = stitched.get_groups_and_index();
  ASSERT_EQ(state.indices.size(), num_rows);
  stitched.sort();
  // Verify sorting correctness
  state = stitched.get_groups_and_index();
  ASSERT_TRUE(is_sorted_by_columns(raw_cols, state.indices));

  // Cleanup
  for (auto* col : raw_cols) {
    delete col;
  }
}

TEST_F(ColumnTest, TwoRoundSimpleSorting) {
  std::vector<uint32_t> col1_data = {1, 2, 3};
  std::vector<uint32_t> col2_data = {4, 5, 6};
  std::vector<uint32_t> col3_data = {7, 8, 9};
  RawColumn raw_col1(col1_data);
  RawColumn raw_col2(col2_data);
  RawColumn raw_col3(col3_data);

  std::vector<RawColumn*> cols = {&raw_col1, &raw_col2};
  std::vector<uint32_t> indices = {0, 1, 2};

  Column first_round = Column::stitch(cols, indices);
  first_round.sort();
  auto state = first_round.get_groups_and_index();
  ASSERT_TRUE(is_sorted_by_columns(cols, state.indices));
  Column second_round = Column::stitch({&raw_col3}, state.indices);
  second_round.sort(state.groups);
  state = second_round.get_groups_and_index();
  ASSERT_TRUE(is_sorted_by_columns({&raw_col3}, state.indices));
}

TEST_F(ColumnTest, TwoRoundSimpleSortingDup) {
  std::vector<uint32_t> col1_data = {1, 2, 2, 1, 1, 4};
  std::vector<uint32_t> col2_data = {4, 2, 2, 4, 1, 4};
  std::vector<uint32_t> col3_data = {6, 9, 8, 5, 4, 3};
  RawColumn raw_col1(col1_data);
  RawColumn raw_col2(col2_data);
  RawColumn raw_col3(col3_data);

  std::vector<RawColumn*> cols = {&raw_col1, &raw_col2};
  std::vector<uint32_t> indices = {0, 1, 2, 3, 4, 5};

  Column first_round = Column::stitch(cols, indices);
  first_round.sort();
  auto state = first_round.get_groups_and_index();
  ASSERT_TRUE(is_sorted_by_columns(cols, state.indices));
  ASSERT_TRUE(state.groups.size() == 4);
  ASSERT_TRUE(state.groups[0].length == 1);
  ASSERT_TRUE(state.groups[1].length == 2);
  ASSERT_TRUE(state.groups[2].length == 2);
  ASSERT_TRUE(state.groups[3].length == 1);
  ASSERT_TRUE(state.indices.size() == 6);
  Column second_round = Column::stitch({&raw_col3}, state.indices);
  second_round.sort(state.groups);
  state = second_round.get_groups_and_index();
  ASSERT_TRUE(
      is_sorted_by_columns({&raw_col1, &raw_col2, &raw_col3}, state.indices));
}

TEST_F(ColumnTest, TwoRoundLargeRandomInput) {
  const size_t num_rows = 1000;
  const size_t num_cols = 4;

  // Generate random data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 100);

  std::vector<std::vector<uint32_t>> col_data(num_cols);
  std::vector<RawColumn*> fstRaw;
  std::vector<RawColumn*> sndRaw;
  std::vector<uint32_t> indices(num_rows);

  // Initialize columns with random data
  col_data[0].resize(num_rows);
  for (size_t j = 0; j < num_rows; j++) {
    col_data[0][j] = dis(gen);
  }
  fstRaw.push_back(new RawColumn(col_data[0]));
  for (size_t i = 1; i < 4; i++) {
    col_data[i].resize(num_rows);
    for (size_t j = 0; j < num_rows; j++) {
      col_data[i][j] = dis(gen);
    }
    sndRaw.push_back(new RawColumn(col_data[i]));
  }

  // Initialize indices
  for (size_t i = 0; i < num_rows; i++) {
    indices[i] = i;
  }

  // Create and sort stitched column
  Column fst = Column::stitch(fstRaw, indices);
  fst.sort();
  auto state = fst.get_groups_and_index();
  ASSERT_EQ(state.indices.size(), num_rows);
  ASSERT_LT(state.groups.size(), num_rows);
  Column snd = Column::stitch(sndRaw, state.indices);
  snd.sort(state.groups);
  state = snd.get_groups_and_index();
  fstRaw.push_back(sndRaw[0]);
  fstRaw.push_back(sndRaw[1]);
  ASSERT_TRUE(is_sorted_by_columns(fstRaw, state.indices));

  // Cleanup
  for (auto* col : fstRaw) {
    delete col;
  }
}