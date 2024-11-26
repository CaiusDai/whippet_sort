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
    12 * 1024 * 1024;            // Various on different CPUs, 12MB here
const size_t CACHE_SIZE = 64;    // 64 bytes cache line
const size_t SCALE_FACTOR = 50;  // How many data exceeds cache size
const size_t NUM_COLUMNS = 4;    // Number of columns
const size_t VALUE_PER_COLUMN = (L3_CACHE_SIZE / 8) * SCALE_FACTOR;
const size_t NUM_RUNS = 5;

typedef vector<vector<int>> StitchPlan;

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
    this->column_count = plan[0].size();
    stitch_timing.resize(plan.size());
    sort_timing.resize(plan.size());
    group_timing.resize(plan.size());
    round_total_timing.resize(plan.size());
  }

  void clear() {
    stitch_timing.clear();
    sort_timing.clear();
    group_timing.clear();
    round_total_timing.clear();
    total_timing.clear();
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

  void write_summary(std::ofstream& output_file) {
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
    for (const auto& column : data) {
      if (column.size() != row_count) {
        std::cerr << "[Error] Data size mismatch: " << column.size() << " vs "
                  << row_count << std::endl;
        return;
      }
    }
    raw_data = data;
  }

  void run_plan(size_t plan_idx, PlanStats& stats, size_t num_runs) {
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
        if (round < round_count - 1) {
          state = std::move(stitched_column.get_groups_and_index());
        } else {
          final_index_list = std::move(stitched_column.get_index_only());
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
    stats.write_summary(output_file);
  }  // function run_plan

  void run_all_plans(size_t num_runs) {
    for (size_t i = 0; i < plans.size(); i++) {
      PlanStats stats(plans[i], raw_data[0].size());
      run_plan(i, stats, num_runs);
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
  Benchmark scatter_bench("benchmark_result_scatter.txt");
  Benchmark centric_bench("benchmark_result_centric.txt");
  scatter_bench.register_plans(plans);
  centric_bench.register_plans(plans);
  std::cout << "Plan Registration Finished\n";
  // Data Registration
  vector<RawColumn> scatter_raw_data, centric_raw_data;
  const size_t row_count = VALUE_PER_COLUMN;
  const size_t column_count = NUM_COLUMNS;
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<uint32_t> dis(0, VALUE_PER_COLUMN / 1000);
  std::uniform_int_distribution<uint32_t> dis_centric(0, 100);
  for (int i = 0; i < column_count; i++) {
    vector<uint32_t> scatter_data(row_count);
    vector<uint32_t> centric_data(row_count);
    for (int j = 0; j < row_count; j++) {
      scatter_data[j] = dis(gen);
      centric_data[j] = dis_centric(gen);
    }
    scatter_raw_data.emplace_back(scatter_data);
    centric_raw_data.emplace_back(centric_data);
  }
  scatter_bench.register_data(scatter_raw_data);
  centric_bench.register_data(centric_raw_data);
  std::cout << "Data Registration Finished\n";

  // Execute benchmark
  scatter_bench.run_all_plans(NUM_RUNS);
  centric_bench.run_all_plans(NUM_RUNS);
}