// The DNest4 sampler (Brewer & Foreman-Mackey 2018) on one generic
// model: coordinates in the unit cube, in a fixed block and in the rows
// of tables whose number of rows varies (an RJObject each: from none to
// the most rows, uniform in number). Its likelihood is a Python function
// of them (which maps them to the free values through the priors).

#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "RJObject/ConditionalPriors/ConditionalPrior.h"
#include "RJObject/RJObject.h"
#include "Sampler.h"
#include "Utils.h"

namespace nb = nanobind;

namespace {

// A table of the model: the coordinates of a row, and the most rows
struct Table
{
    int columns;
    int rows;
};

// The shape of the models of the running sampler, and their likelihood:
// DNest4 makes its models with their default constructor, which reads it
struct Shape
{
    int fixed;
    std::vector<Table> tables;
    nb::callable log_likelihood;
};

// (one per thread: DNest4 runs in the thread that calls run, and samplers
// may run in several threads at once)
thread_local const Shape* shape = nullptr;

// The prior of the rows of the tables: uniform in the unit cube, as the
// coordinates of the RJObject are
class UnitPrior : public DNest4::ConditionalPrior
{
    double perturb_hyperparameters(DNest4::RNG&) override { return 0.0; }

public:
    void from_prior(DNest4::RNG&) override {}
    double log_pdf(const std::vector<double>&) const override { return 0.0; }
    void from_uniform(std::vector<double>&) const override {}
    void to_uniform(std::vector<double>&) const override {}
    void print(std::ostream&) const override {}
};

// Return a numpy array of the given values and shape (it owns them)
nb::object
to_numpy(std::vector<double> values, std::initializer_list<size_t> dims)
{
    auto* owned = new std::vector<double>(std::move(values));
    nb::capsule owner(owned, [](void* p) noexcept {
        delete static_cast<std::vector<double>*>(p);
    });
    return nb::cast(nb::ndarray<nb::numpy, double>(owned->data(), dims, owner));
}

// Return a numpy array of the given rows (rows, columns)
nb::object
to_numpy(const std::vector<std::vector<double>>& rows, size_t columns)
{
    std::vector<double> values;
    values.reserve(rows.size() * columns);
    for (const auto& row : rows)
        values.insert(values.end(), row.begin(), row.end());
    return to_numpy(std::move(values), {rows.size(), columns});
}

class Model
{
public:
    Model()
        : fixed_(shape->fixed)
    {
        for (const Table& table : shape->tables)
            tables_.emplace_back(
                table.columns, table.rows, false, UnitPrior(),
                DNest4::PriorType::uniform);
    }

    void from_prior(DNest4::RNG& rng)
    {
        for (double& u : fixed_)
            u = rng.rand();
        for (auto& table : tables_)
            table.from_prior(rng);
    }

    double perturb(DNest4::RNG& rng)
    {
        // The tables 3 times in 4 (as in the examples of DNest4), else
        // the fixed block
        if (tables_.empty() || (!fixed_.empty() && rng.rand() > 0.75))
            return DNest4::perturb_us(fixed_, rng);
        return tables_[rng.rand_int(tables_.size())].perturb(rng, false);
    }

    double log_likelihood() const
    {
        nb::list rows;
        for (size_t i = 0; i < tables_.size(); ++i)
            rows.append(to_numpy(
                tables_[i].get_components(), shape->tables[i].columns));
        return nb::cast<double>(shape->log_likelihood(
            to_numpy(fixed_, {fixed_.size()}), rows));
    }

    // The fixed block, then for each table its number of rows and the
    // coordinates of all its rows (0 in those it does not use)
    void print(std::ostream& out) const
    {
        out << std::setprecision(17);
        for (double u : fixed_)
            out << u << ' ';
        for (size_t i = 0; i < tables_.size(); ++i) {
            const auto& rows = tables_[i].get_components();
            const Table& table = shape->tables[i];
            out << rows.size() << ' ';
            for (const auto& row : rows)
                for (double u : row)
                    out << u << ' ';
            for (size_t j = rows.size() * table.columns;
                 j < size_t(table.rows * table.columns); ++j)
                out << 0.0 << ' ';
        }
    }

    std::string description() const
    {
        return "the fixed coordinates, then for each table its number of "
               "rows and the coordinates of all its rows";
    }

private:
    std::vector<double> fixed_;
    std::vector<DNest4::RJObject<UnitPrior>> tables_;
};

// Discards what is written to it
class NullBuffer : public std::streambuf
{
protected:
    int overflow(int c) override { return c; }
};

// Silences std::cout while any sampler runs (DNest4 reports its progress
// there): std::cout is shared by the threads, so the first run silences
// it and the last restores it (each restoring what it found would restore
// the buffer of a run that ended first, if the runs end in another order)
class Silence
{
public:
    Silence()
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (runs_++ == 0)
            old_ = std::cout.rdbuf(&buffer_);
    }

    ~Silence()
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (--runs_ == 0)
            std::cout.rdbuf(old_);
    }

private:
    static inline std::mutex mutex_;
    static inline int runs_ = 0;
    static inline std::streambuf* old_ = nullptr;
    static inline NullBuffer buffer_;
};

// Clears the shape of the running sampler when it ends (or raises)
struct ShapeReset
{
    ~ShapeReset() { shape = nullptr; }
};

void
run(int fixed,
    const std::vector<int>& columns,
    const std::vector<int>& rows,
    const nb::callable& log_likelihood,
    unsigned int particles,
    unsigned int new_level_interval,
    unsigned int save_interval,
    unsigned int thread_steps,
    int max_levels,
    double lambda,
    double beta,
    unsigned int max_saves,
    unsigned int seed,
    const std::string& sample_file,
    const std::string& sample_info_file,
    const std::string& levels_file)
{
    if (shape)
        throw std::runtime_error("DNest4 is already running in this thread");
    if (fixed < 0 || columns.size() != rows.size())
        throw std::invalid_argument("invalid shape of the model");
    for (size_t i = 0; i < columns.size(); ++i)
        if (columns[i] < 1 || rows[i] < 1)
            throw std::invalid_argument("invalid shape of a table");
    if (fixed == 0 && columns.empty())
        throw std::invalid_argument("the model has no coordinates");
    if (particles < 1 || new_level_interval < 1 || thread_steps < 1
        || !(lambda > 0) || !(beta >= 0))
        throw std::invalid_argument("invalid options of DNest4");

    Shape current{fixed, {}, log_likelihood};
    for (size_t i = 0; i < columns.size(); ++i)
        current.tables.push_back({columns[i], rows[i]});
    ShapeReset reset;
    shape = &current;

    DNest4::Options options(
        particles, new_level_interval, save_interval, thread_steps,
        max_levels, lambda, beta, max_saves);
    options.set_files(sample_file, sample_info_file, levels_file);
    Silence silence;
    // (one thread, and the compression of automatic levels: e)
    DNest4::Sampler<Model> sampler(1, std::exp(1.0), options, true);
    sampler.initialise(seed);
    sampler.run();
}

} // namespace

NB_MODULE(EXTENSION_NAME, m)
{
    m.def("run", &run,
          nb::arg("fixed"), nb::arg("columns"), nb::arg("rows"),
          nb::arg("log_likelihood"), nb::kw_only(),
          nb::arg("particles"), nb::arg("new_level_interval"),
          nb::arg("save_interval"), nb::arg("thread_steps"),
          nb::arg("max_levels"), nb::arg("lambda_"), nb::arg("beta"),
          nb::arg("max_saves"), nb::arg("seed"), nb::arg("sample_file"),
          nb::arg("sample_info_file"), nb::arg("levels_file"),
          "Run DNest4 on the model of the given shape: fixed coordinates, "
          "and tables of the given columns and most rows. It calls "
          "log_likelihood(fixed, rows) with the coordinates in the unit "
          "cube: an array of the fixed ones, and a list of an array (rows, "
          "columns) per table. It writes the samples, their information "
          "and the levels to the given files (see DNest4).");
}
