// dogm_rfs.hpp - corrected particle-based Dynamic Occupancy Grid Map.
//
// ============================================================================
// REFERENCES  (see REFERENCES.md)
//   [1] D. Nuss, S. Reuter, M. Thom, T. Yuan, G. Krehl, M. Maile, A. Gern,
//       K. Dietmayer, "A Random Finite Set Approach for Dynamic Occupancy Grid
//       Maps with Real-Time Application", Int. J. Robotics Research 37(8),
//       2018. arXiv:1605.02406.
//       -> prediction WITH process noise, occupancy update, persistent/newborn
//          particle split, and global resampling. This file follows its
//          predict / update / resample / birth cycle.
//   [2] R. Danescu, F. Oniga, S. Nedevschi, "Modeling and Tracking the Driving
//       Environment With a Particle-Based Occupancy Grid", IEEE T-ITS 12(4),
//       2011. -> particle-based occupancy grid carrying per-particle velocity.
//   [3] S. Steyer, G. Tanzmeister, D. Wollherr, "Object tracking based on
//       evidential dynamic occupancy grids in urban environments", IEEE IV,
//       2017. -> track -> grid feedback, consumed here through `seed_vel`.
//   [3b] G. Chen, W. Dong, P. Peng, J. Alonso-Mora, X. Zhu, "Continuous
//       Occupancy Mapping in Dynamic Environments Using Particles", IEEE
//       T-RO, 2024. arXiv:2202.06273.
//       -> newborn particles get an INITIAL VELOCITY ESTIMATE rather than a
//          blind prior, and birth draws from a MIXTURE of a static model and a
//          constant-velocity model. Both are adopted below.
//   [4] G. Grisetti, C. Stachniss, W. Burgard, "Improved Techniques for Grid
//       Mapping with Rao-Blackwellized Particle Filters", IEEE T-RO 23(1),
//       2007. -> adaptive resampling on the effective sample size, and an
//       informed proposal that folds in the most recent information rather
//       than sampling blindly from the motion model.
//
// ============================================================================
// WHY THIS FILE EXISTS
//
// The original inline implementation (legacy/dgm_track_final.cpp) could not
// estimate velocity even in principle. Three defects had to hold simultaneously
// for that to be true, and fixing any one alone changes nothing measurable -
// which is why the earlier tuning effort could not converge:
//
//   B1  The measurement update was a mathematical no-op. Every particle in a
//       cell was multiplied by the SAME likelihood ratio (computed from the
//       cell occupancy, not from the particle) and the cell was then
//       renormalised to sum to 1. Normalising a uniform rescale reproduces the
//       input exactly, so posterior == prior. The cross-cell weight
//       differences, which are the only channel through which velocity
//       evidence can enter, were destroyed by that per-cell normalisation.
//   B2  Prediction never perturbed particle velocities (it only ego-rotated
//       them), so a particle's velocity was frozen at birth and the filter
//       could never explore a velocity it had not been born with.
//   B3  There was no resampling anywhere, so weight differences never turned
//       into population differences - and population is what actually encodes
//       the posterior in this filter family.
//   B4  Birth velocities were N(seed, 2 m/s) about a seed taken from the track
//       estimate, which cannot represent fast objects and is self-referential
//       (the track velocity was itself initialised from the DOGM velocity), so
//       no external evidence ever entered the loop.
//
// Measured consequence on nuScenes val (150 scenes, 103k matched GT boxes):
// the legacy DOGM velocity had EPE 1.085 m/s against GT while the raw detector
// velocity had 0.298 m/s, and the legacy DOGM was also ~1.8x more jittery than
// the detector it was meant to stabilise.
//
// ============================================================================
// SCOPE NOTE
//
// Particles are allocated over detection-box regions rather than the whole
// grid. A full-grid allocation at 0.2 m resolution would need millions of
// particles for the same per-object density, and this system only ever reads
// per-object velocity. The filter mechanics (predict / weight / resample /
// birth) remain those of [1]; only the support region is restricted. Cells
// outside any box therefore act as negative evidence, which is what removes
// particles whose velocity carried them off the object.
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>
#ifdef DOGM_PROFILE
#include <chrono>
#include <cstdio>
#endif

#include <Eigen/Dense>

namespace dogm {

// ---------------------------------------------------------------- grid spec
struct GridSpec {
    double resolution = 0.2;
    double lookforward = 50.0;
    double lookback = 50.0;
    double lookside = 50.0;
    int W = 500, H = 500;

    void finalize() {
        W = int((lookside / resolution) * 2);
        H = int((lookforward + lookback) / resolution);
    }
    // std::lround without the library call: round half away from zero. For
    // |v| < 2^30 the cast truncates exactly and v - trunc(v) is exact, so the
    // result equals std::lround; anything else (NaN, huge) takes std::lround.
    static inline int round_cell(double v) {
        if (!(std::fabs(v) < 1073741824.0)) return (int)std::lround(v);
        long long i = (long long)v;
        const double d = v - (double)i;
        if (d >= 0.5) ++i;
        else if (d <= -0.5) --i;
        return (int)i;
    }
    inline int px(double x) const { return round_cell((x + lookside) / resolution); }
    inline int py(double y) const { return round_cell((lookforward - y) / resolution); }
    inline double x_of(int p) const { return (p * resolution) - lookside; }
    inline double y_of(int p) const { return lookforward - (p * resolution); }
    inline bool inside(int p, int q) const {
        return (unsigned)p < (unsigned)W && (unsigned)q < (unsigned)H;
    }
    inline int index(int p, int q) const { return q * W + p; }
};

struct BoxLidar2D {
    double cx = 0, cy = 0;   // center, LiDAR frame (m)
    double w = 0, l = 0;     // width (across), length (along heading), m
    double yaw = 0;          // rad
};

struct BoxVelocity {
    Eigen::Vector2d v = Eigen::Vector2d::Zero();
    Eigen::Matrix2d S = Eigen::Matrix2d::Identity() * 1e6;
    double conf = 0.0;
    double mass = 0.0;
    int n_particles = 0;
    int n_cells = 0;
    bool valid = false;
};

struct Config {
    GridSpec grid;

    // particle budget is per detection box, not per cell: only per-object
    // velocity is ever read out, and a box needs ~10^3 samples to represent a
    // 2-D velocity distribution regardless of how many cells it covers.
    int particles_per_box = 400;
    int max_particles = 300000;

    // [1] prediction process noise (fix B2).
    // Sized against the velocity resolution of a single step: with dt ~ 0.05 s
    // (20 Hz sweeps) and 0.2 m cells, one step can only separate velocities
    // more than 0.2/0.05 = 4 m/s apart, so evidence has to accumulate over
    // ~10 frames. Diffusion large enough to jump a cell per step would destroy
    // that accumulation faster than it builds up.
    double sigma_v_proc = 0.3;   // m/s per sqrt(s)
    double sigma_p_proc = 0.02;  // m per sqrt(s)

    // measurement model
    double occ_measured = 0.9;    // occupancy credited to a cell with returns
    double occ_unmeasured = 0.1;  // negative evidence elsewhere (fix B1 needs
                                  // this to be < occ_measured to discriminate)
    int meas_dilate = 1;          // cells; tolerates box/point misalignment

    // [1],[4] resampling (fix B3).
    // Resampling every frame would reset the weights before a single step's
    // weak evidence could accumulate, so it is triggered adaptively on the
    // effective sample size instead ([4], adaptive resampling).
    bool resample = true;
    double resample_neff_ratio = 0.5;  // resample when N_eff < ratio * N
    // Birth counts each box's current population from the cell buckets. The
    // original code (default, false) uses the buckets built BEFORE resampling,
    // whose indices point at other particles once resampling has reordered the
    // population, so under-counted boxes receive extra births and the population
    // grows well past particles_per_box. true re-buckets after resampling so the
    // count is correct. Changes the results; kept off for every reported run.
    bool fresh_birth_count = false;
    // Resampling keeps the population size N, so weight moving between boxes
    // never shrinks it: depleted boxes get births and N only grows (measured:
    // ~1,500 particles per box on nuScenes, ~800 on KITTI, for a budget of 400).
    // true resamples to min(N, particles_per_box x boxes) instead. Changes the
    // results; off for every reported run.
    bool resample_budget = false;

    // birth velocity prior (fix B4).
    // Birth is targeted: particles are only injected for boxes whose current
    // population is too small to support a velocity read-out (new or re-entering
    // objects). Blanket rebirth every frame would keep flooding well-tracked
    // objects with wide-prior noise.
    double birth_vel_sigma = 6.0;
    double birth_vel_max = 25.0;
    double birth_mass_share = 0.05;  // total weight given to newborns per frame
    double prune_pad_m = 0.5;        // keep particles within this margin of a box
    // Share of newborns drawn around an external seed velocity (track
    // feedback, [3]); this is an informed proposal in the sense of [4], using
    // the most recent object-level estimate instead of a blind prior. The
    // remainder always uses the wide prior so evidence can still overrule the
    // tracker - that is what prevents the self-referential lock-in which
    // disabled the original filter (B4).
    double birth_seed_fraction = 0.5;
    double birth_seed_sigma = 2.0;
    // Explicit static hypothesis. [1] separates static and dynamic occupancy
    // mass; without it every newborn carries the wide dynamic prior, and a
    // parked car's posterior mean never collapses to zero because the evidence
    // over a few frames cannot rule out the slow hypotheses it was born with.
    // Measured effect of that omission: EPE 0.406 m/s on static objects versus
    // 0.107 for the raw detector, over 59% of all matched boxes.
    // Birth model [3b]: a mixture of a static model and a constant-velocity
    // model around an INITIAL VELOCITY ESTIMATE, instead of a blind wide prior.
    //
    // Why this matters here. With a blind N(0, 6 m/s) prior the posterior never
    // leaves the prior, because "is the particle inside the detection box" is
    // an almost uninformative measurement: a 4.5 m box against 0.5 m of travel
    // per 20 Hz sweep rewards a wide band of velocities equally. The measured
    // symptom was a magnitude collapse - reported speed 0.6x truth for fast
    // objects, and |mean| pinned at the Monte-Carlo noise floor
    // (sigma/sqrt(N_eff) ~ 0.6 m/s) for parked cars, i.e. an 8x over-estimate
    // of a near-zero speed - while the direction stayed accurate to ~5 deg.
    //
    // Seeding births at an external estimate concentrates the prior where the
    // answer actually is, so the grid evidence refines it instead of having to
    // discover it. This is what makes the DOGM a cheap refinement stage rather
    // than a competitor to the detector's own velocity head.
    double birth_static_fraction = 0.10;   // share drawn from the static model
    double birth_static_sigma_v = 0.10;    // m/s, spread of the static model
    // Share of the dynamic births drawn around the supplied initial estimate;
    // the remainder keeps the wide prior so the grid can still contradict it.
    double birth_init_fraction = 0.70;
    double birth_init_sigma = 1.5;         // m/s around the initial estimate
    // |v_init| below this is treated as a static measurement, and then ALL of
    // that box's births use the static model [3b].
    double static_speed_thresh = 0.5;
    double birth_static_sigma = 0.15;  // m/s, near-zero but not degenerate

    // velocity read-out
    double sigma_v_floor = 0.35;
    double min_mass_for_velocity = 1e-9;
    int min_particles_for_velocity = 8;
    // Maturity gate. A freshly seeded box still carries the wide birth prior,
    // so its posterior mean is meaningless until the evidence has narrowed the
    // spread. `valid == false` tells the fusion stage to fall back to the
    // detector.
    //
    // The threshold must scale with speed. A fixed absolute limit rejects fast
    // objects - whose velocity posterior is legitimately wider - and so admits
    // only the narrow, slow-biased estimates. Measured effect of the fixed
    // 2.5 m/s version: reported speed saturated at ~2.5 m/s (GT 12.7 m/s came
    // back as 2.55, ratio 0.20) while the direction stayed accurate to 5 deg,
    // i.e. a pure magnitude collapse produced by the gate's selection bias.
    double vel_std_abs = 2.5;   // m/s, floor of the allowed spread
    double vel_std_rel = 0.6;   // plus this fraction of the estimated speed

    uint64_t seed = 12345;  // fix B7: deterministic by default
};

struct Particle {
    float x = 0, y = 0;
    float vx = 0, vy = 0;
    float w = 0;
};

// std::normal_distribution<float> driven by std::mt19937, written out: the
// Marsaglia polar method of libstdc++ with generate_canonical<float, 24>, the
// same float/double promotions and the same saved second value, so it returns
// bit-identical draws in the same order (checked against libstdc++ 15.2 on
// 2e7 draws) while skipping the long-double path of generate_canonical.
// ~20% cheaper per draw; the predict step makes four draws per particle.
class NormalF {
public:
    NormalF(float mean, float stddev) : mean_(mean), stddev_(stddev) {}
    inline float operator()(std::mt19937& g) {
        float ret;
        if (avail_) {
            avail_ = false;
            ret = saved_;
        } else {
            float x, y, r2;
            do {
                x = 2.0f * canon(g) - 1.0;
                y = 2.0f * canon(g) - 1.0;
                r2 = x * x + y * y;
            } while (r2 > 1.0 || r2 == 0.0);
            const float mult = std::sqrt(-2 * std::log(r2) / r2);
            saved_ = x * mult;
            avail_ = true;
            ret = y * mult;
        }
        return ret * stddev_ + mean_;
    }

private:
    static inline float canon(std::mt19937& g) {
        float r = (float)(g() - std::mt19937::min()) / 4294967296.0f;
        if (r >= 1.0f) r = 0.99999994f;  // nextafter(1.f, 0.f), as generate_canonical
        return r;
    }
    float mean_, stddev_;
    bool avail_ = false;
    float saved_ = 0.0f;
};

// ============================================================================
// SPEED NOTE (2026-09-25)
//
// The filter below computes exactly what the pre-2026-09-25 version did
// (src/legacy/dogm_rfs_before_speedup_20260925.hpp): the same arithmetic in
// the same order and the same random draws in the same order, so every output
// is bit-identical (checked by output hash on nuScenes and KITTI). Only the
// bookkeeping changed:
//   * cell -> occupancy and cell -> particles were std::unordered_map; they are
//     now dense per-cell arrays reset through a list of touched cells, and the
//     particles of a cell are kept in ascending index order (a stable counting
//     sort), which is the order the old per-cell vectors were filled in;
//   * the cos/sin of each box are computed once per step instead of once per
//     point-in-box test;
//   * each particle's cell is computed once, in predict(), and carried through
//     update / bucket / prune / resample instead of being recomputed four times;
//     GridSpec::px/py round with an inline equivalent of std::lround;
//   * the four normal draws per particle use NormalF, a written-out copy of
//     libstdc++'s std::normal_distribution<float> (same draws, same order);
//   * work buffers are reused across steps instead of being reallocated.
// Same-condition A/B, nuScenes (2 scenes): 41.7 -> 22.5 ms per step. The
// remaining cost is mostly the random draws of predict(), which fix the
// results and so stay. Two result-changing options exist, both off by default
// and not adopted (PREREGISTRATION 2026-09-25): fresh_birth_count and
// resample_budget.
// Kept on purpose: birth() counts the population of each box with the cell
// buckets built before resampling, as the old code did; changing that would
// change the results.
// ============================================================================
class DogmRfs {
public:
    explicit DogmRfs(const Config& cfg) : cfg_(cfg), rng_(cfg.seed) {
        cfg_.grid.finalize();
        const size_t ncell = (size_t)cfg_.grid.W * cfg_.grid.H;
        occ_on_.assign(ncell, 0);
        bucket_cnt_.assign(ncell, 0);
        bucket_off_.assign(ncell, 0);
    }

    void reset() {
        parts_.clear();
        pcell_.clear();
        clear_occ();
        frame_ = 0;
    }

    const Config& config() const { return cfg_; }
    size_t particle_count() const { return parts_.size(); }

    // Fraction of a query box's footprint that received a LiDAR return this
    // frame. `hit_` already holds every return, not only those inside a
    // detection, so this answers "is something still there?" for a region the
    // detector produced no box for - which is what a coasting track needs.
    // Returns -1 when the box lies outside the grid.
    double box_support(const BoxLidar2D& b) const {
        const auto& g = cfg_.grid;
        const CellRange r = box_cell_range(b, 0);
        int total = 0, hit = 0;
        for (int y = r.y0; y <= r.y1; ++y)
            for (int x = r.x0; x <= r.x1; ++x) {
                if (!point_in_box(g.x_of(x), g.y_of(y), b)) continue;
                ++total;
                if (!hit_.empty() && hit_[(size_t)g.index(x, y)]) ++hit;
            }
        return total ? (double)hit / total : -1.0;
    }

    // One filter cycle. `R_rel`/`t_rel` map the previous LiDAR frame into the
    // current one; `dt` is the elapsed time; `points` are the current returns;
    // `boxes` are the detections to report velocity for; `seed_vel` is an
    // optional per-box track velocity (empty disables the feedback).
    void step(const Eigen::Matrix2d& R_rel, const Eigen::Vector2d& t_rel, double dt,
              const std::vector<Eigen::Vector2f>& points,
              const std::vector<BoxLidar2D>& boxes,
              const std::vector<Eigen::Vector2d>& seed_vel,
              std::vector<BoxVelocity>& out) {
#ifdef DOGM_PROFILE
        // per-stage time, printed when the filter is destroyed (profiling builds only)
        auto T = [](){ return std::chrono::steady_clock::now(); };
        auto t = T();
        auto lap = [&](int k) { const auto n = T(); prof_[k] += std::chrono::duration<double, std::milli>(n - t).count(); t = n; };
#else
        auto lap = [](int) {};
#endif
        geo_.resize(boxes.size());
        for (size_t b = 0; b < boxes.size(); ++b) geo_[b] = geom(boxes[b]);
        n_boxes_ = (int)boxes.size();
        predict(R_rel, t_rel, dt);            lap(0);
        build_measurement(points, boxes);     lap(1);
        update_weights();                     lap(2);
        bucket_particles();                   lap(3);
        extract_box_velocity(boxes, out);     lap(4);  // reads the weighted posterior
        prune_unsupported(boxes);             lap(5);
        bucket_particles();                   lap(3);
        const bool resampled = cfg_.resample && maybe_resample();
        if (resampled && cfg_.fresh_birth_count) bucket_particles();
        lap(6);
        birth(boxes, seed_vel);               lap(7);
        ++frame_;
    }

#ifdef DOGM_PROFILE
    ~DogmRfs() {
        const char* nm[8] = {"predict", "measurement", "update", "bucket", "extract", "prune", "resample", "birth"};
        double tot = 0;
        for (double v : prof_) tot += v;
        std::fprintf(stderr, "[dogm profile] %d steps, %.1f ms total\n", frame_, tot);
        for (int k = 0; k < 8; ++k)
            std::fprintf(stderr, "  %-12s %8.1f ms  %5.1f%%  %.3f ms/step\n", nm[k], prof_[k],
                         tot > 0 ? 100.0 * prof_[k] / tot : 0.0, frame_ ? prof_[k] / frame_ : 0.0);
    }
#endif

private:
    // A box with its trigonometry evaluated once. in_box() is point_in_box()
    // with cos/sin and the half-extent limits hoisted; same operations, same
    // values.
    struct BoxG { double cx, cy, c, s, lim_l, lim_w; };
    static BoxG geom(const BoxLidar2D& b) {
        BoxG q;
        q.cx = b.cx;
        q.cy = b.cy;
        q.c = std::cos(b.yaw);
        q.s = std::sin(b.yaw);
        q.lim_l = b.l * 0.5 + 1e-3;
        q.lim_w = b.w * 0.5 + 1e-3;
        return q;
    }
    static inline bool in_box(double x, double y, const BoxG& b) {
        const double dx = x - b.cx, dy = y - b.cy;
        const double lx = b.c * dx + b.s * dy, ly = -b.s * dx + b.c * dy;
        return std::fabs(lx) <= b.lim_l && std::fabs(ly) <= b.lim_w;
    }

    // ---------------------------------------------------------------- predict
    void predict(const Eigen::Matrix2d& R_rel, const Eigen::Vector2d& t_rel, double dt) {
        if (parts_.empty()) return;
        const float sv = (float)(cfg_.sigma_v_proc * std::sqrt(std::max(1e-3, dt)));
        const float sp = (float)(cfg_.sigma_p_proc * std::sqrt(std::max(1e-3, dt)));
        NormalF nv(0.0f, sv), np(0.0f, sp);  // = std::normal_distribution<float>, bit for bit

        std::vector<Particle>& kept = tmp_;
        kept.clear();
        kept.reserve(parts_.size());
        pcell_.clear();
        pcell_.reserve(parts_.size());
        const auto& g = cfg_.grid;
        for (const auto& p : parts_) {
            Eigen::Vector2d pos(p.x, p.y), vel(p.vx, p.vy);
            pos = R_rel * pos + t_rel;
            vel = R_rel * vel;

            // fix B2 - without diffusion the velocity posterior can only
            // collapse, never track a changing velocity.
            Particle q;
            q.vx = (float)vel.x() + nv(rng_);
            q.vy = (float)vel.y() + nv(rng_);
            q.x = (float)(pos.x() + q.vx * dt) + np(rng_);
            q.y = (float)(pos.y() + q.vy * dt) + np(rng_);
            q.w = p.w;

            const int cx = g.px(q.x), cy = g.py(q.y);
            if (!g.inside(cx, cy)) continue;
            kept.push_back(q);
            pcell_.push_back(g.index(cx, cy));  // the particle's cell, reused below
        }
        parts_.swap(kept);
    }

public:
    static bool point_in_box(double x, double y, const BoxLidar2D& b) {
        const double c = std::cos(b.yaw), s = std::sin(b.yaw);
        const double dx = x - b.cx, dy = y - b.cy;
        const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
        return std::fabs(lx) <= b.l * 0.5 + 1e-3 && std::fabs(ly) <= b.w * 0.5 + 1e-3;
    }

    // Cell-index range covering a box's axis-aligned bounding box, padded.
    struct CellRange { int x0, x1, y0, y1; };
    CellRange box_cell_range(const BoxLidar2D& b, int pad) const {
        const auto& g = cfg_.grid;
        const double r = 0.5 * std::hypot(b.l, b.w);
        int x0 = g.px(b.cx - r) - pad, x1 = g.px(b.cx + r) + pad;
        int y0 = g.py(b.cy + r) - pad, y1 = g.py(b.cy - r) + pad;  // py inverts y
        return {std::max(0, std::min(x0, x1)), std::min(g.W - 1, std::max(x0, x1)),
                std::max(0, std::min(y0, y1)), std::min(g.H - 1, std::max(y0, y1))};
    }

private:
    // --------------------------------------------------------- cell bookkeeping
    void clear_occ() {
        for (int c : occ_cells_) occ_on_[(size_t)c] = 0;
        occ_cells_.clear();
    }
    inline void set_occ(int c) {
        if (!occ_on_[(size_t)c]) {
            occ_on_[(size_t)c] = 1;
            occ_cells_.push_back(c);
        }
    }

    // ------------------------------------------------------------ measurement
    // Occupancy evidence = LiDAR returns that fall inside a detection box.
    // Points are binned once (O(N)); boxes then only scan their own bounding
    // box of cells, instead of the legacy O(boxes x points) full rescan.
    void build_measurement(const std::vector<Eigen::Vector2f>& points,
                           const std::vector<BoxLidar2D>& boxes) {
        const auto& g = cfg_.grid;
        hit_.assign((size_t)g.W * g.H, 0);
        for (const auto& p : points) {
            const int cx = g.px(p.x()), cy = g.py(p.y());
            if (g.inside(cx, cy)) hit_[(size_t)g.index(cx, cy)] = 1;
        }

        clear_occ();
        for (size_t b = 0; b < boxes.size(); ++b) {
            const BoxG& bg = geo_[b];
            const CellRange r = box_cell_range(boxes[b], cfg_.meas_dilate);
            for (int y = r.y0; y <= r.y1; ++y) {
                for (int x = r.x0; x <= r.x1; ++x) {
                    if (!hit_[(size_t)g.index(x, y)]) continue;
                    if (!in_box(g.x_of(x), g.y_of(y), bg)) continue;
                    // dilate so a particle just off the measured cell is not
                    // treated as being in free space
                    for (int dy = -cfg_.meas_dilate; dy <= cfg_.meas_dilate; ++dy)
                        for (int dx = -cfg_.meas_dilate; dx <= cfg_.meas_dilate; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (g.inside(nx, ny)) set_occ(g.index(nx, ny));
                        }
                }
            }
        }
    }

    // ----------------------------------------------------------------- update
    // fix B1: per-particle likelihood from the cell it actually landed in, and
    // a GLOBAL normalisation that preserves cross-cell weight differences.
    void update_weights() {
        if (parts_.empty()) return;
        const auto& g = cfg_.grid;
        const double lr_free = cfg_.occ_unmeasured / std::max(1e-9, 1.0 - cfg_.occ_unmeasured);
        // every measured cell holds (float)occ_measured, so its ratio is one value
        const double o = std::clamp((double)(float)cfg_.occ_measured, 1e-3, 1.0 - 1e-3);
        const double lr_occ = o / (1.0 - o);

        // every particle is inside the grid here (predict dropped the others)
        // and pcell_ holds its cell
        (void)g;
        double wsum = 0.0;
        for (size_t i = 0; i < parts_.size(); ++i) {
            Particle& p = parts_[i];
            const double lr = occ_on_[(size_t)pcell_[i]] ? lr_occ : lr_free;
            p.w = (float)(p.w * lr);
            wsum += p.w;
        }
        if (wsum > 1e-12) {
            const float inv = (float)(1.0 / wsum);
            for (auto& p : parts_) p.w *= inv;
        } else {
            const float u = 1.0f / (float)parts_.size();
            for (auto& p : parts_) p.w = u;
        }
    }

    // Particles grouped by cell (compressed rows): bucket_cnt_[c] particles of
    // cell c, their indices at bucket_idx_[bucket_off_[c] ...], ascending.
    // Uses pcell_, which predict / prune / resample keep aligned with parts_.
    void bucket_particles() {
        for (int c : bucket_cells_) bucket_cnt_[(size_t)c] = 0;
        bucket_cells_.clear();
        for (size_t i = 0; i < parts_.size(); ++i) {
            const int c = pcell_[i];
            if (bucket_cnt_[(size_t)c]++ == 0) bucket_cells_.push_back(c);
        }
        int off = 0;
        for (int c : bucket_cells_) {
            bucket_off_[(size_t)c] = off;
            off += bucket_cnt_[(size_t)c];
        }
        bucket_idx_.resize((size_t)off);
        for (size_t i = 0; i < parts_.size(); ++i) {
            const int c = pcell_[i];
            bucket_idx_[(size_t)bucket_off_[(size_t)c]++] = (int)i;
        }
        for (int c : bucket_cells_) bucket_off_[(size_t)c] -= bucket_cnt_[(size_t)c];
    }

    // Drop particles that have drifted off every detection box. Their velocity
    // hypothesis is exactly the one the free-space evidence just rejected, and
    // keeping them would let the population grow until the budget is exhausted,
    // starving newly appearing objects of births.
    // Uses the cell buckets so the cost is proportional to the boxes' own
    // footprints, not to (particles x boxes).
    void prune_unsupported(const std::vector<BoxLidar2D>& boxes) {
        if (parts_.empty()) return;
        if (boxes.empty()) { parts_.clear(); pcell_.clear(); return; }
        const auto& g = cfg_.grid;
        const double pad = cfg_.prune_pad_m;
        const int pad_cells = std::max(1, (int)std::ceil(pad / g.resolution));

        keep_.assign(parts_.size(), 0);
        for (const auto& b : boxes) {
            BoxLidar2D e = b;
            e.w += 2 * pad;
            e.l += 2 * pad;
            const BoxG eg = geom(e);
            const CellRange r = box_cell_range(b, pad_cells);
            for (int y = r.y0; y <= r.y1; ++y)
                for (int x = r.x0; x <= r.x1; ++x) {
                    const size_t c = (size_t)g.index(x, y);
                    const int n = bucket_cnt_[c];
                    if (!n) continue;
                    const int* q = &bucket_idx_[(size_t)bucket_off_[c]];
                    for (int k = 0; k < n; ++k) {
                        const int idx = q[k];
                        if (!keep_[(size_t)idx] && in_box(parts_[idx].x, parts_[idx].y, eg))
                            keep_[(size_t)idx] = 1;
                    }
                }
        }
        std::vector<Particle>& kept = tmp_;
        kept.clear();
        kept.reserve(parts_.size());
        size_t m = 0;
        for (size_t i = 0; i < parts_.size(); ++i)
            if (keep_[i]) {
                kept.push_back(parts_[i]);
                pcell_[m++] = pcell_[i];
            }
        pcell_.resize(m);
        parts_.swap(kept);
        double s = 0.0;
        for (const auto& p : parts_) s += p.w;
        if (s > 1e-12) {
            const float inv = (float)(1.0 / s);
            for (auto& p : parts_) p.w *= inv;
        }
    }

    // ------------------------------------------------------- velocity read-out
    void extract_box_velocity(const std::vector<BoxLidar2D>& boxes,
                              std::vector<BoxVelocity>& out) {
        out.assign(boxes.size(), BoxVelocity{});
        if (parts_.empty()) return;
        const auto& g = cfg_.grid;

        for (size_t b = 0; b < boxes.size(); ++b) {
            const BoxG& bg = geo_[b];
            double w = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            int n = 0, ncell = 0;
            const CellRange r = box_cell_range(boxes[b], 0);
            for (int y = r.y0; y <= r.y1; ++y) {
                for (int x = r.x0; x <= r.x1; ++x) {
                    const size_t c = (size_t)g.index(x, y);
                    const int m = bucket_cnt_[c];
                    if (!m) continue;
                    const int* q = &bucket_idx_[(size_t)bucket_off_[c]];
                    bool used = false;
                    for (int k = 0; k < m; ++k) {
                        const Particle& p = parts_[q[k]];
                        if (p.w <= 0 || !in_box(p.x, p.y, bg)) continue;
                        w += p.w;
                        sx += p.w * p.vx;  sy += p.w * p.vy;
                        sxx += p.w * (double)p.vx * p.vx;
                        syy += p.w * (double)p.vy * p.vy;
                        sxy += p.w * (double)p.vx * p.vy;
                        ++n;
                        used = true;
                    }
                    if (used) ++ncell;
                }
            }

            BoxVelocity& o = out[b];
            o.mass = w;
            o.n_particles = n;
            o.n_cells = ncell;
            if (w <= cfg_.min_mass_for_velocity || n < cfg_.min_particles_for_velocity)
                continue;

            const double mx = sx / w, my = sy / w;
            double vxx = std::max(0.0, sxx / w - mx * mx);
            double vyy = std::max(0.0, syy / w - my * my);
            const double vxy = sxy / w - mx * my;
            vxx += cfg_.sigma_v_floor * cfg_.sigma_v_floor;
            vyy += cfg_.sigma_v_floor * cfg_.sigma_v_floor;

            o.v << mx, my;
            o.S << vxx, vxy, vxy, vyy;
            const double spd = std::hypot(mx, my);
            const double allow = cfg_.vel_std_abs + cfg_.vel_std_rel * spd;
            o.valid = std::sqrt(std::max(vxx, vyy)) <= allow;

            // confidence: how far the velocity is from zero relative to its own
            // spread (chi-square, 2 dof), scaled by the supporting population.
            Eigen::LLT<Eigen::Matrix2d> llt(o.S);
            const double chi2 = (llt.info() == Eigen::Success)
                                    ? (double)(o.v.transpose() * llt.solve(o.v))
                                    : 0.0;
            const double c_motion = 1.0 - std::exp(-0.5 * std::max(0.0, chi2));
            const double c_support =
                std::min(1.0, (double)n / std::max(1, cfg_.particles_per_box / 4));
            o.conf = c_motion * std::sqrt(std::max(0.0, c_support));
        }
    }

    // ----------------------------------------------------- adaptive resampling
    // [1],[4] fix B3. Triggered on the effective sample size rather than every
    // frame: one 20 Hz step carries very little velocity evidence, so the
    // weights must be allowed to accumulate across frames before the
    // population is redrawn.
    bool maybe_resample() {
        if (parts_.empty()) return false;
        double w2 = 0.0;
        for (const auto& p : parts_) w2 += (double)p.w * p.w;
        if (w2 <= 0) return false;
        const double neff = 1.0 / w2;  // weights are normalised
        if (neff >= cfg_.resample_neff_ratio * parts_.size()) return false;

        int n = (int)parts_.size();
        if (cfg_.resample_budget && n_boxes_ > 0)
            n = std::min(n, cfg_.particles_per_box * n_boxes_);
        std::vector<Particle>& next = tmp_;
        next.clear();
        next.reserve(n);
        std::vector<int>& next_cell = tmp_cell_;
        next_cell.clear();
        next_cell.reserve(n);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double step = 1.0 / n;
        double target = u01(rng_) * step;
        double c = parts_[0].w;
        size_t i = 0;
        for (int k = 0; k < n; ++k) {
            while (c < target && i + 1 < parts_.size()) c += parts_[++i].w;
            Particle q = parts_[i];
            q.w = (float)step;
            next.push_back(q);
            next_cell.push_back(pcell_[i]);
            target += step;
        }
        parts_.swap(next);
        pcell_.swap(next_cell);
        ++n_resample_;
        return true;
    }

    // ---------------------------------------------------------- targeted birth
    // Only boxes that cannot currently support a velocity read-out receive new
    // particles, so established objects are not repeatedly flooded with
    // wide-prior noise.
    void birth(const std::vector<BoxLidar2D>& boxes,
               const std::vector<Eigen::Vector2d>& seed_vel) {
        if (boxes.empty()) return;
        const auto& g = cfg_.grid;
        const int want = cfg_.particles_per_box;
        const int floor_n = std::max(cfg_.min_particles_for_velocity * 2, want / 4);

        // current population per box (buckets of the pre-resampling population,
        // as in the original implementation - see SPEED NOTE)
        have_.assign(boxes.size(), 0);
        for (size_t b = 0; b < boxes.size(); ++b) {
            const BoxG& bg = geo_[b];
            const CellRange r = box_cell_range(boxes[b], 0);
            for (int y = r.y0; y <= r.y1; ++y)
                for (int x = r.x0; x <= r.x1; ++x) {
                    const size_t c = (size_t)g.index(x, y);
                    const int m = bucket_cnt_[c];
                    if (!m) continue;
                    const int* q = &bucket_idx_[(size_t)bucket_off_[c]];
                    for (int k = 0; k < m; ++k)
                        if (in_box(parts_[q[k]].x, parts_[q[k]].y, bg)) ++have_[b];
                }
        }

        std::uniform_real_distribution<float> jit(-0.5f, 0.5f);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        std::normal_distribution<float> wide(0.0f, (float)cfg_.birth_vel_sigma);
        std::normal_distribution<float> near_init(0.0f, (float)cfg_.birth_init_sigma);
        std::normal_distribution<float> stat_n(0.0f, (float)cfg_.birth_static_sigma_v);
        const float vmax = (float)cfg_.birth_vel_max;

        std::vector<Particle>& born = born_;
        born.clear();
        std::vector<int>& cells = cells_;
        for (size_t b = 0; b < boxes.size(); ++b) {
            if (have_[b] >= floor_n) continue;
            int need = want - have_[b];
            if ((int)(parts_.size() + born.size()) + need > cfg_.max_particles)
                need = cfg_.max_particles - (int)(parts_.size() + born.size());
            if (need <= 0) continue;

            cells.clear();
            const BoxG& bg = geo_[b];
            const CellRange r = box_cell_range(boxes[b], 0);
            for (int y = r.y0; y <= r.y1; ++y)
                for (int x = r.x0; x <= r.x1; ++x) {
                    if (!occ_on_[(size_t)g.index(x, y)]) continue;
                    if (!in_box(g.x_of(x), g.y_of(y), bg)) continue;
                    cells.push_back(g.index(x, y));
                }
            if (cells.empty()) continue;
            std::uniform_int_distribution<size_t> pick(0, cells.size() - 1);

            // [3b] initial velocity estimate for this box, if the caller gave one
            const bool has_init = b < seed_vel.size();
            const Eigen::Vector2d v0 = has_init ? seed_vel[b] : Eigen::Vector2d::Zero();
            // A measurement whose initial estimate is essentially at rest is
            // treated as static, and then every birth uses the static model.
            const bool meas_static =
                has_init && v0.norm() < cfg_.static_speed_thresh;

            for (int k = 0; k < need; ++k) {
                const int cell = cells[pick(rng_)];
                Particle q;
                q.x = (float)(g.x_of(cell % g.W) + jit(rng_) * g.resolution);
                q.y = (float)(g.y_of(cell / g.W) + jit(rng_) * g.resolution);

                const double u = u01(rng_);
                if (meas_static || u < cfg_.birth_static_fraction) {
                    // static model [3b]
                    q.vx = stat_n(rng_);
                    q.vy = stat_n(rng_);
                } else if (has_init && u < cfg_.birth_static_fraction +
                                              (1.0 - cfg_.birth_static_fraction) *
                                                  cfg_.birth_init_fraction) {
                    // constant-velocity model about the initial estimate [3b]
                    q.vx = (float)v0.x() + near_init(rng_);
                    q.vy = (float)v0.y() + near_init(rng_);
                } else {
                    // wide prior, so evidence can still overrule the estimate
                    q.vx = wide(rng_);
                    q.vy = wide(rng_);
                }
                q.vx = std::clamp(q.vx, -vmax, vmax);
                q.vy = std::clamp(q.vy, -vmax, vmax);
                born.push_back(q);
            }
        }
        if (born.empty()) return;

        // Newborns collectively receive `birth_mass_share` of the total weight;
        // the persistent set keeps the rest, so a burst of births cannot
        // overwhelm accumulated evidence.
        const float bw = (float)(cfg_.birth_mass_share / born.size());
        const float keep = (float)(1.0 - cfg_.birth_mass_share);
        for (auto& p : parts_) p.w *= keep;
        for (auto& q : born) { q.w = bw; parts_.push_back(q); }

        double s = 0.0;
        for (const auto& p : parts_) s += p.w;
        if (s > 1e-12) {
            const float inv = (float)(1.0 / s);
            for (auto& p : parts_) p.w *= inv;
        }
    }

    Config cfg_;
    std::mt19937 rng_;
    std::vector<Particle> parts_;
    std::vector<uint8_t> hit_;
    int frame_ = 0;
    int n_resample_ = 0;

    // per-step boxes with their trigonometry
    std::vector<BoxG> geo_;
    int n_boxes_ = 0;
    // measured cells: dense flag per cell + the list of set cells (for reset)
    std::vector<uint8_t> occ_on_;
    std::vector<int> occ_cells_;
    // cell index of each particle, aligned with parts_ from predict() until
    // birth() appends (newborns get theirs at the next predict())
    std::vector<int> pcell_, tmp_cell_;
    // particles by cell (compressed rows)
    std::vector<int> bucket_cnt_, bucket_off_, bucket_cells_, bucket_idx_;
    // reused work buffers
    std::vector<Particle> tmp_, born_;
    std::vector<uint8_t> keep_;
    std::vector<int> have_, cells_;
#ifdef DOGM_PROFILE
    double prof_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#endif
};

}  // namespace dogm
