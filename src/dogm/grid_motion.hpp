// grid_motion.hpp - lightweight per-object grid motion cue.
//
// ============================================================================
// WHAT THIS IS
//
// For each tracked object, the LiDAR returns that fall inside its box are
// rasterised onto a global occupancy lattice. The object's velocity is then the
// lattice displacement that best aligns its occupancy pattern with the pattern
// stored for the same track one keyframe earlier, divided by the elapsed time.
//
// No particles, no per-cell filter: one small binary patch and a bounded
// correlation search per object. That is the entire method.
//
// ============================================================================
// WHY IT REPLACED THE PARTICLE DOGM HERE
//
// The particle filter was measured to carry no usable velocity information in
// this setting, and the reason is a resolution argument, not an implementation
// detail. Particles were reweighted by whether they landed in a cell with a
// return inside the detection box. Between 20 Hz sweeps an object travelling
// 10 m/s moves 0.5 m, against a ~4.5 m box: a wide band of velocities keeps a
// particle inside the box and therefore earns the same reward. Measured
// consequence - the posterior never left the prior, so reported speed came back
// at 0.6x truth for fast objects and at the Monte-Carlo noise floor of the mean
// magnitude for parked ones, while the direction stayed accurate.
//
// Correlating across a KEYFRAME baseline instead inverts that ratio. Over 0.5 s
// the same object moves 5 m = 25 cells at 0.2 m resolution, so the displacement
// is large compared to the cell and the correlation peak is well localised.
// The velocity quantum becomes resolution/dt = 0.4 m/s before sub-cell
// refinement, against 4 m/s per sweep - a 10x improvement from timing alone.
//
// This is also why the cue is genuinely independent of the detector: the
// correlation is computed on raw returns, whereas the detector velocity comes
// from box regression. Two different failure modes, which is what makes fusing
// them worthwhile.
//
// ============================================================================
// REFERENCES (see REFERENCES.md)
//   [25] Optical Flow Based Detection and Tracking of Moving Objects for
//        Autonomous Vehicles, 2024. arXiv:2403.17779 - converts LiDAR scans to
//        a motion grid and estimates a 2-D velocity per cell by comparing
//        consecutive scans.
//   [26] A. Dewan, T. Caselitz, G. D. Tipaldi, W. Burgard, "Motion-based
//        Detection and Tracking in 3D LiDAR Scans", ICRA 2016.
//   [27] Motion Estimation in Occupancy Grid Maps in Stationary Settings,
//        2019. arXiv:1909.11387.
//   [3b] G. Chen et al., "Continuous Occupancy Mapping in Dynamic Environments
//        Using Particles", IEEE T-RO 2024. arXiv:2202.06273 - the static /
//        constant-velocity mixture and the idea of seeding from an initial
//        velocity estimate, reused here as the search prior.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

namespace gm {

struct Config {
    double resolution = 0.2;      // m per lattice cell
    // Half-width of the displacement search. It bounds the detectable speed at
    // search_radius_m / dt, so 3 m over a 0.5 s baseline would silently clip
    // everything above 6 m/s - which showed up as the >8 m/s band collapsing to
    // 0.55 of true speed. Sized here for ~20 m/s.
    double search_radius_m = 10.0;
    double dt_min = 0.2;          // s; below this the baseline is too short
    double dt_max = 1.2;          // s; above this the patch has changed too much
    int min_cells = 25;           // patches smaller than this are not matched
    double min_peak_score = 0.25; // normalised overlap required at the peak
    // A peak only marginally better than the runner-up is ambiguous (a smooth
    // or repetitive surface); confidence is scaled by the margin.
    double min_peak_margin = 0.02;
    bool subcell = true;          // parabolic sub-cell refinement
};

struct Estimate {
    Eigen::Vector2d v = Eigen::Vector2d::Zero();  // m/s, global frame
    double conf = 0.0;                            // [0,1]
    double peak = 0.0;                            // normalised overlap at peak
    int n_cells = 0;
    bool valid = false;
};

// Occupancy footprint of one object, as absolute lattice cells.
struct Patch {
    double t = 0.0;
    std::vector<int32_t> cx, cy;
};

class GridMotion {
public:
    explicit GridMotion(const Config& cfg) : cfg_(cfg) {}

    void reset() { prev_.clear(); accum_.clear(); }
    size_t tracked() const { return prev_.size(); }

    // Accumulate one sweep's worth of returns for a track, motion-compensated
    // to the reference time `t_ref` using the track's current velocity.
    //
    // This is what makes the cue usable at all. A single sweep leaves only the
    // visible surface of an object - measured mean 77 occupied cells, and
    // fewer than `min_cells` for most objects, so the correlation fired on
    // only 5.8% of tracks. The ten unlabelled sweeps between keyframes see the
    // same object from ten slightly different viewpoints; de-rotating each by
    // the ego pose and de-translating by the object's own motion stacks them
    // into one dense patch without smearing it.
    void accumulate(int track_id, double t_ref, double t_sweep,
                    const std::vector<Eigen::Vector2d>& pts_global,
                    const Eigen::Vector2d& v_global) {
        auto& a = accum_[track_id];
        const double dt = t_sweep - t_ref;
        for (const auto& q : pts_global) a.push_back(q - v_global * dt);
        if (a.size() > 20000) a.resize(20000);
    }

    void clear_accum(int track_id) { accum_.erase(track_id); }

    // Match using the accumulated points instead of a single sweep.
    Estimate estimate_accum(int track_id, double t_now,
                            const Eigen::Vector2d& prior_disp) {
        auto it = accum_.find(track_id);
        static const std::vector<Eigen::Vector2d> kEmpty;
        Estimate e = estimate(track_id, t_now,
                              it == accum_.end() ? kEmpty : it->second, prior_disp);
        accum_.erase(track_id);
        return e;
    }

    // Rasterise `pts` (global XY, already restricted to the object) and match
    // against this track's previous patch. `prior_disp` centres the search -
    // pass the detector's predicted displacement when available, zero
    // otherwise; it only bounds the search window, it does not bias the peak.
    Estimate estimate(int track_id, double t_now,
                      const std::vector<Eigen::Vector2d>& pts,
                      const Eigen::Vector2d& prior_disp) {
        Estimate out;
        Patch cur;
        cur.t = t_now;
        rasterise(pts, cur);
        out.n_cells = (int)cur.cx.size();

        auto it = prev_.find(track_id);
        if (it == prev_.end() || (int)cur.cx.size() < cfg_.min_cells) {
            if ((int)cur.cx.size() >= cfg_.min_cells) prev_[track_id] = std::move(cur);
            return out;
        }

        const Patch& old = it->second;
        const double dt = t_now - old.t;
        if (dt < cfg_.dt_min || dt > cfg_.dt_max ||
            (int)old.cx.size() < cfg_.min_cells) {
            prev_[track_id] = std::move(cur);
            return out;
        }

        // current occupancy as a lookup set
        occ_.clear();
        occ_.reserve(cur.cx.size() * 2);
        for (size_t i = 0; i < cur.cx.size(); ++i)
            occ_.insert(key(cur.cx[i], cur.cy[i]));

        const int R = std::max(1, (int)std::lround(cfg_.search_radius_m / cfg_.resolution));
        const int px = (int)std::lround(prior_disp.x() / cfg_.resolution);
        const int py = (int)std::lround(prior_disp.y() / cfg_.resolution);
        const double norm = std::sqrt((double)old.cx.size() * (double)cur.cx.size());

        double best = -1.0, second = -1.0;
        int bx = 0, by = 0;
        scores_.assign((size_t)(2 * R + 1) * (2 * R + 1), 0.0f);
        for (int dy = -R; dy <= R; ++dy) {
            for (int dx = -R; dx <= R; ++dx) {
                const int sx = px + dx, sy = py + dy;
                int hit = 0;
                for (size_t i = 0; i < old.cx.size(); ++i)
                    if (occ_.count(key(old.cx[i] + sx, old.cy[i] + sy))) ++hit;
                const double sc = hit / norm;
                scores_[(size_t)(dy + R) * (2 * R + 1) + (dx + R)] = (float)sc;
                if (sc > best) { second = best; best = sc; bx = sx; by = sy; }
                else if (sc > second) second = sc;
            }
        }

        prev_[track_id] = std::move(cur);
        out.peak = best;
        if (best < cfg_.min_peak_score) return out;

        double fx = bx, fy = by;
        if (cfg_.subcell) refine(R, px, py, bx, by, fx, fy);

        out.v = Eigen::Vector2d(fx * cfg_.resolution / dt, fy * cfg_.resolution / dt);
        // Confidence combines how much of the surface re-appeared with how
        // unambiguous the peak was; a flat correlation surface means the shape
        // could not localise the shift (a long wall, a smooth side panel).
        const double margin = std::max(0.0, best - std::max(0.0, second));
        out.conf = std::clamp(best, 0.0, 1.0) *
                   std::clamp(margin / std::max(cfg_.min_peak_margin, 1e-6), 0.0, 1.0);
        out.valid = out.conf > 0.0;
        return out;
    }

    void forget(int track_id) { prev_.erase(track_id); }

private:
    static inline int64_t key(int32_t x, int32_t y) {
        return ((int64_t)x << 32) ^ (uint32_t)y;
    }

    void rasterise(const std::vector<Eigen::Vector2d>& pts, Patch& p) const {
        static thread_local std::unordered_set<int64_t> seen;
        seen.clear();
        p.cx.clear();
        p.cy.clear();
        for (const auto& q : pts) {
            const int32_t ix = (int32_t)std::floor(q.x() / cfg_.resolution);
            const int32_t iy = (int32_t)std::floor(q.y() / cfg_.resolution);
            if (seen.insert(key(ix, iy)).second) {
                p.cx.push_back(ix);
                p.cy.push_back(iy);
            }
        }
    }

    // Parabolic interpolation through the peak and its two neighbours per axis.
    void refine(int R, int px, int py, int bx, int by, double& fx, double& fy) const {
        const int W = 2 * R + 1;
        const int ix = bx - px + R, iy = by - py + R;
        auto at = [&](int x, int y) -> double {
            if (x < 0 || x >= W || y < 0 || y >= W) return 0.0;
            return scores_[(size_t)y * W + x];
        };
        const double c = at(ix, iy);
        const double l = at(ix - 1, iy), r = at(ix + 1, iy);
        const double d = at(ix, iy - 1), u = at(ix, iy + 1);
        const double denx = (l - 2 * c + r), deny = (d - 2 * c + u);
        if (std::fabs(denx) > 1e-9) fx = bx + std::clamp(0.5 * (l - r) / denx, -0.5, 0.5);
        if (std::fabs(deny) > 1e-9) fy = by + std::clamp(0.5 * (d - u) / deny, -0.5, 0.5);
    }

    Config cfg_;
    std::unordered_map<int, Patch> prev_;
    std::unordered_map<int, std::vector<Eigen::Vector2d>> accum_;
    std::unordered_set<int64_t> occ_;
    std::vector<float> scores_;
};

}  // namespace gm
