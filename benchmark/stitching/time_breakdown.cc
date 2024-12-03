#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "column.h"
#include "timer.h"
using std::string;
using std::vector;
namespace stitch {

const size_t L3_CACHE_SIZE =
    12 * 1024 * 1024;           // Various on different CPUs, 12MB here
const size_t CACHE_SIZE = 64;   // 64 bytes cache line
const size_t SCALE_FACTOR = 1;  // How many data exceeds cache size
const size_t NUM_COLUMNS = 4;   // Number of columns
const size_t VALUE_PER_COLUMN = (L3_CACHE_SIZE / 8) * SCALE_FACTOR;
const size_t NUM_RUNS = 5;

typedef vector<vector<int>> StitchPlan;

class Generator {
 public:
  Generator(size_t row_count, size_t column_count, double cardinality_rate)
      : row_count(row_count), column_count(column_count) {
    if (cardinality_rate <= 0 || cardinality_rate > 1) {
      throw std::runtime_error("[ERROR] Invalid cardinality rate");
    }
  }
  vector<RawColumn> generate();
  // Fields
  size_t row_count;
  size_t column_count;
  double cardinality_rate;
};

vector<RawColumn> Generator::generate() {
  uint32_t lower_bound = 0;
  uint32_t upper_bound = row_count * cardinality_rate;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(lower_bound, upper_bound);
  vector<RawColumn> raw_data;
  for (size_t i = 0; i < column_count; i++) {
    vector<uint32_t> column_data(row_count);
    for (size_t j = 0; j < row_count; j++) {
      column_data[j] = dis(gen);
    }
    raw_data.emplace_back(column_data);
  }
  return raw_data;
}

class PlanStats {
 public:
  enum class TimingType { STITCH, SORT, GROUP, ROUND };
  PlanStats(StitchPlan plan, size_t row_count) {
    if (plan.size() < 1) {
      std::cerr << "[Error] Invalid plan size " << plan.size() << std::endl;
      return;
    }
    this->plan = plan;
    this->row_count = row_count;
    this->column_count = 0;
    for (const auto& round : plan) {
      this->column_count += round.size();
    }
    stitch_timing.resize(plan.size());
    sort_timing.resize(plan.size());
    group_timing.resize(plan.size());
    round_total_timing.resize(plan.size());
    unique_group_counts.resize(plan.size());
  }

  void clear() {
    stitch_timing.clear();
    sort_timing.clear();
    group_timing.clear();
    round_total_timing.clear();
    total_timing.clear();
    unique_group_counts.clear();
  }

  void set_group_count(size_t round, uint64_t count) {
    unique_group_counts[round] = count;
  }

  size_t compute_skipped_data_rate() {
    double result = 0;
    size_t prev_rounds_sum = 0;
    uint64_t total_data = row_count * column_count;
    uint64_t saved_data = 0;
    uint64_t stitched_column = 0;
    for (size_t i = 0; i < plan.size() - 1; i++) {
      stitched_column += plan[i].size();
      saved_data += (unique_group_counts[i] - prev_rounds_sum) *
                    (column_count - stitched_column);
      prev_rounds_sum = unique_group_counts[i];
    }
    return (saved_data * 100 / total_data);
  }

  inline void record(TimingType type, size_t round, double time) {
    switch (type) {
      case TimingType::STITCH:
        stitch_timing[round].push_back(time);
        break;
      case TimingType::SORT:
        sort_timing[round].push_back(time);
        break;
      case TimingType::GROUP:
        group_timing[round].push_back(time);
        break;
      case TimingType::ROUND:
        round_total_timing[round].push_back(time);
        break;
      default:
        throw std::runtime_error("[ERROR] Invalid timing type");
    }
  }

  inline void record_total(double time) { total_timing.push_back(time); }

  inline double get_median(const vector<double>& timing) {
    if (timing.size() < 1) {
      return 0;
    }
    vector<double> sorted_timing = timing;
    std::sort(sorted_timing.begin(), sorted_timing.end());
    return sorted_timing[timing.size() / 2];
  }

  void write_summary(std::ofstream& output_file, bool write_group = false) {
    if (!output_file.is_open()) {
      throw std::runtime_error("[ERROR] Output file is not open");
    }
    output_file << "Plan: ";
    for (const auto& round : plan) {
      output_file << "[";
      for (const auto& column : round) {
        output_file << column << ",";
      }
      output_file << "] ";
    }
    output_file << std::endl;
    output_file << "Row count: " << row_count << std::endl;
    output_file << "Column count: " << column_count << std::endl;
    if (write_group) {
      output_file << "Skipped data rate: " << compute_skipped_data_rate() << "%"
                  << std::endl;
      output_file << "Unique group counts: \n";
      for (int i = 0; i < unique_group_counts.size(); i++) {
        output_file << "[Round " << i << "] " << unique_group_counts[i] << "/"
                    << row_count << "\n";
      }
    }
    output_file << "Total time: " << get_median(total_timing) << "ms"
                << std::endl;
    for (int i = 0; i < plan.size(); i++) {
      output_file << "Round " << i
                  << " : Stitch: " << get_median(stitch_timing[i]) << "ms, "
                  << "Sort: " << get_median(sort_timing[i]) << "ms, "
                  << "Group: " << get_median(group_timing[i]) << "ms, "
                  << "Total: " << get_median(round_total_timing[i]) << "ms"
                  << std::endl;
    }
    output_file << std::endl;
  }

 private:
  StitchPlan plan;
  size_t row_count;
  size_t column_count;
  vector<vector<double>> stitch_timing;
  vector<vector<double>> sort_timing;
  vector<vector<double>> group_timing;
  vector<vector<double>> round_total_timing;
  vector<double> total_timing;
  vector<uint64_t> unique_group_counts;
};

class Benchmark {
 public:
  Benchmark(const string& file_path) {
    output_file = std::ofstream(file_path);
    if (!output_file.is_open()) {
      string error_msg = "Failed to open output file: " + file_path;
      throw std::runtime_error(error_msg);
    }
  }

  ~Benchmark() { output_file.close(); }

  inline void register_plan(const StitchPlan& plan) { plans.push_back(plan); }
  inline void register_plans(const vector<StitchPlan>& plans) {
    for (const auto& plan : plans) {
      register_plan(plan);
    }
  }

  inline void register_data(const vector<RawColumn>& data) {
    if (data.size() < 1) {
      std::cerr << "[Error] Invalid data size " << data.size() << std::endl;
      return;
    }
    const int row_count = data[0].size();
    // Check if all columns have the same size of data
    for (const auto& column : data) {
      if (column.size() != row_count) {
        std::cerr << "[Error] Data size mismatch: " << column.size() << " vs "
                  << row_count << std::endl;
        return;
      }
    }
    raw_data = data;
  }

  void collect_group_info(size_t plan_idx, PlanStats& stats) {
    if (plan_idx >= plans.size()) {
      std::cerr << "[Error] Invalid plan index " << plan_idx
                << ", plan size: " << plans.size() << std::endl;
      return;
    }
    const StitchPlan& plan = plans[plan_idx];
    const size_t round_count = plan.size();
    vector<RawColumn*> columns;  // Buffer for each round
    SortingState state;
    vector<uint32_t> final_index_list;
    size_t row_count = raw_data[0].size();
    state.indices.reserve(row_count);
    for (size_t i = 0; i < row_count; i++) {
      state.indices.push_back(i);
    }

    // Execute each round
    for (size_t round = 0; round < round_count; round++) {
      const vector<int>& stitch_columns = plan[round];
      columns.clear();
      for (const auto& column_idx : stitch_columns) {
        columns.push_back(&raw_data[column_idx]);
      }

      // Stitching & Possible Permutation
      Column stitched_column = Column::stitch(columns, state.indices);

      // Sorting
      if (round == 0) {
        // First round
        stitched_column.sort();
      } else {
        // Rest of sorting
        stitched_column.sort(state.groups);
      }

      // Grouping Lookup
      if (round == 0) {
        state = stitched_column.get_groups_and_index();
      } else if (round < round_count - 1) {
        state = stitched_column.get_groups_and_index(state.groups);
      } else {
        final_index_list = stitched_column.get_index_only();
      }
      size_t unique_group_count = 0;
      for (const auto& group : state.groups) {
        if (group.length == 1) {
          unique_group_count++;
        }
      }
      stats.set_group_count(round, unique_group_count);
    }  // For loop for each round
  }

  void run_plan(size_t plan_idx, PlanStats& stats, size_t num_runs,
                bool write_group = false) {
    if (plan_idx >= plans.size()) {
      std::cerr << "[Error] Invalid plan index " << plan_idx
                << ", plan size: " << plans.size() << std::endl;
      return;
    }
    const StitchPlan& plan = plans[plan_idx];
    const size_t round_count = plan.size();

    // @var Global: Total time to execute a plan
    // @var Round: Time to execute each round
    // @var Operator: Time to execute STITCH,SORT and GROUP operators. Reused
    // for each Op.
    Timer global_timer, operator_timer, round_timer;

    for (size_t run = 0; run < num_runs; run++) {
      vector<RawColumn*> columns;  // Buffer for each round
      SortingState state;
      vector<uint32_t> final_index_list;
      // Time counting start here
      global_timer.start();

      size_t row_count = raw_data[0].size();
      state.indices.reserve(row_count);
      for (size_t i = 0; i < row_count; i++) {
        state.indices.push_back(i);
      }

      // Execute each round
      for (size_t round = 0; round < round_count; round++) {
        round_timer.start();
        const vector<int>& stitch_columns = plan[round];
        columns.clear();
        for (const auto& column_idx : stitch_columns) {
          columns.push_back(&raw_data[column_idx]);
        }

        // Stitching & Possible Permutation
        operator_timer.start();
        Column stitched_column = Column::stitch(columns, state.indices);
        operator_timer.stop();
        stats.record(PlanStats::TimingType::STITCH, round,
                     operator_timer.get_elapsed_time_ms());

        // Sorting
        operator_timer.start();
        if (round == 0) {
          // First round
          stitched_column.sort();
        } else {
          // Rest of sorting
          stitched_column.sort(state.groups);
        }
        operator_timer.stop();
        stats.record(PlanStats::TimingType::SORT, round,
                     operator_timer.get_elapsed_time_ms());

        // Grouping Lookup
        operator_timer.start();
        if (round == 0) {
          state = stitched_column.get_groups_and_index();
        } else if (round < round_count - 1) {
          state = stitched_column.get_groups_and_index(state.groups);
        } else {
          final_index_list = stitched_column.get_index_only();
        }
        operator_timer.stop();
        stats.record(PlanStats::TimingType::GROUP, round,
                     operator_timer.get_elapsed_time_ms());
        round_timer.stop();
        stats.record(PlanStats::TimingType::ROUND, round,
                     round_timer.get_elapsed_time_ms());
      }  // For loop for each round
      global_timer.stop();
      stats.record_total(global_timer.get_elapsed_time_ms());
    }  // For loop for repeated runs
    stats.write_summary(output_file, write_group);
  }  // function run_plan

  void run_all_plans(size_t num_runs) {
    for (size_t i = 0; i < plans.size(); i++) {
      PlanStats stats(plans[i], raw_data[0].size());
      // Collect group info
      collect_group_info(i, stats);
      run_plan(i, stats, num_runs, true);
    }
  }

 private:
  vector<StitchPlan> plans;
  vector<RawColumn> raw_data;
  std::ofstream output_file;
};
}  // namespace stitch

int main() {
  using namespace stitch;
  // Plan Registration
  vector<StitchPlan> plans;
  plans.push_back({{0, 1, 2, 3}});
  plans.push_back({{0, 1}, {2}, {3}});
  plans.push_back({{0, 1}, {2, 3}});
  plans.push_back({{0}, {1, 2}, {3}});
  plans.push_back({{0}, {1}, {2, 3}});
  plans.push_back({{0, 1, 2}, {3}});
  plans.push_back({{0}, {1, 2, 3}});
  plans.push_back({{0}, {1}, {2}, {3}});
  vector<double> group_setting = {0.2, 0.4, 0.6, 0.8, 1.0};
  // Data Registration
  vector<RawColumn> raw_data;
  const size_t row_count = VALUE_PER_COLUMN;
  const size_t column_count = NUM_COLUMNS;
  Generator generator(row_count, column_count, 0.5);
  for (int i = 0; i < group_setting.size(); i++) {
    std::cout << "[INFO] Executing for cardinality rate: " << group_setting[i]
              << std::endl;
    generator.cardinality_rate = group_setting[i];
    raw_data = generator.generate();
    Benchmark benchmark(std::string("benchmark_result_" +
                                    std::to_string(group_setting[i]) + ".txt"));
    benchmark.register_plans(plans);
    benchmark.register_data(raw_data);
    std::cout << "[INFO] Registration finished\n";
    benchmark.run_all_plans(NUM_RUNS);
  }
}