#pragma once
#include <fstream>
#include <string>
#include <iomanip>

// =========================================================
// CSV HEADER
// =========================================================
inline void csv_header(const std::string& filename) {
    std::ofstream f(filename, std::ios::out);
    f << "algorithm,variant,device,precision,size,"
         "time_ms,gflops,speedup,rmse,abs_error,rel_error\n";
    f.close();
}

// =========================================================
// CSV ADD ROW
// =========================================================
inline void csv_add(const std::string& filename,
                    const std::string& algorithm,
                    const std::string& variant,
                    const std::string& device,
                    const std::string& precision,
                    int size,
                    double time_ms,
                    double gflops,
                    double speedup,
                    double rmse,
                    double abs_error,
                    double rel_error) {

    std::ofstream f(filename, std::ios::app);

    f << algorithm << ","
      << variant << ","
      << device << ","
      << precision << ","
      << size << ","
      << time_ms << ","
      << gflops << ","
      << speedup << ","
      << rmse << ","
      << abs_error << ","
      << rel_error << "\n";

    f.close();
}