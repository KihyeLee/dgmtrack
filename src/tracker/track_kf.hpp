// track_kf.hpp - constant-velocity Kalman tracker with pluggable velocity fusion.
//
// Its role in this project is not only to produce track IDs. The DOGM's own
// velocity read-out is the mean of a particle cloud, which carries Monte-Carlo
// noise that is redrawn every frame and shows up directly as jitter - the very
// quantity this project wants to minimise. Feeding those read-outs into a
// per-object Kalman filter replaces the sample mean with an analytic recursive
// estimate (Rao-Blackwellisation): the noise is averaged out instead of being
// resampled, and the estimate becomes smooth by construction.
//
// That requires object identity across frames, which a standalone DOGM does not
// have - its particle population is its only cross-frame memory. Association is
// what supplies it, which is why coupling the tracker to the DOGM improves the
// velocity estimate rather than merely consuming it.
//
// References (see REFERENCES.md):
//   [8] Steyer et al., IEEE IV 2017 - track -> grid feedback, association that
//       accounts for the velocity difference
//   [9]  Steyer et al., IEEE ITSC 2019 - object-level -> grid-level backchannel
//   [21] OptiPMB, 2025 - RFS track lifecycle (adaptive birth, adaptive
//        detection probability under occlusion); the reference for replacing
//        the hits/miss heuristics below with a principled lifecycle
//   [23] Julier & Uhlmann, ACC 1997 - Covariance Intersection, used by
//        Fusion::Ci for the closed loop where measurement and prior are
//        correlated by construction

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace trk {

// How a velocity measurement is merged into a track.
enum class Fusion {
    None,      // ignore it; velocity only evolves through the motion model
    Replace,   // overwrite (what a naive "use the DOGM velocity" does)
    Kf,        // standard Kalman update with the measurement covariance
    Ci,        // Covariance Intersection - conservative when the measurement
               // and the prior are correlated, which they are here because the
               // DOGM was seeded from this very track
};

struct VelMeas {
    Eigen::Vector2d v = Eigen::Vector2d::Zero();
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * 1e6;
    double conf = 0.0;
    bool valid = false;
    // Per-measurement innovation gate, in chi-square units with 2 degrees of
    // freedom. <= 0 means "use Config::chi2_gate_vel".
    //
    // Why this is per-measurement rather than a constant. Measured on the dev
    // set, the detector's velocity error at a low detection score is not more
    // Gaussian-noisy - the median error is flat across the whole score range
    // (0.123 m/s at score 0.21 against 0.110 m/s at 0.89, a factor of 1.1).
    // What changes is the tail: p99 rises 6.2x (7.19 against 1.16 m/s) and the
    // probability of an error above 2 m/s rises 11.9x (5.5% against 0.46%).
    //
    // That distinction decides the mechanism. Inflating R, which is what
    // confidence-adaptive trackers do, models a wider Gaussian: it damps the
    // 95% of low-score detections that were perfectly good and still lets a
    // 7 m/s outlier move the state. Tightening the innovation gate instead
    // rejects the outlier outright and leaves the good detections at full
    // weight. The threshold is set from the measured outlier rate rather than
    // tuned - see tools/calibrate_noise.py.
    double chi2_gate = 0.0;
};

struct Config {
    // motion model
    double q_pos = 0.05;    // m^2/s
    double q_vel = 0.5;     // (m/s)^2/s
    double r_pos = 1.0;     // m, position measurement std
    // Optional per-class overrides (index = nusc::class_id). Empty vectors keep
    // the global values above, so the default behaviour is unchanged. This is
    // the class-wise Q/R baseline that a grid velocity cue has to beat.
    std::vector<double> q_vel_by_cls, r_pos_by_cls;
    double q_vel_for(int cls) const {
        return (cls >= 0 && cls < (int)q_vel_by_cls.size()) ? q_vel_by_cls[cls] : q_vel;
    }
    double r_pos_for(int cls) const {
        return (cls >= 0 && cls < (int)r_pos_by_cls.size()) ? r_pos_by_cls[cls] : r_pos;
    }

    // association
    double gate_maha2 = 9.0;   // ~3 sigma on position
    bool class_gate = true;

    // lifecycle.
    // Coasting is measured in SECONDS, not in frames. The same frame count
    // means 0.5 s when the tracker steps on 20 Hz sweeps and 10 s when it
    // steps on 2 Hz keyframes, so a frame-based limit silently changes the
    // tracker's behaviour with the stepping mode and makes runs
    // incomparable. The original code used max_miss = 10 frames, which was
    // 0.5 s on sweeps against the legacy tracker's effective 10 s - the
    // dominant cause of its 3x worse fragmentation.
    double max_coast_s = 2.0;
    int confirm_hits = 2;

    // velocity fusion
    Fusion fusion = Fusion::Kf;
    double sigma_v_floor = 0.5;   // m/s, added to every measurement covariance
    double conf_min = 0.05;       // below this a measurement is ignored
    // Age-gated use of an external velocity measurement. Calibration against
    // ground truth (docs/GRID_VELOCITY_CALIBRATION.md) shows the grid velocity
    // beats the track's own constant-velocity estimate only while the track is
    // young (fewer than ~3 keyframes of history) and moving; afterwards the
    // filter's own estimate is better and the measurement only perturbs it.
    // <= 0 disables the gate (every matched keyframe may use the measurement).
    int vel_max_hits = 0;
    // Time-based version of the same gate (exploratory, PREREGISTRATION 2026-09-24 18:00): the velocity
    // measurement is used only while the track is at most this many seconds old. <= 0 disables it.
    double vel_max_age_s = 0.0;
    // Confidence a measurement needs to seed a new track's velocity. < 0 keeps
    // the original behaviour (seed whenever the measurement is valid), so the
    // birth, age and update-confidence rules can be ablated independently.
    double seed_conf_min = -1.0;
    double chi2_gate_vel = 9.0;   // reject wild velocity measurements
    // Calibration of the velocity measurement noise (PREREGISTRATION 2026-09-24, experiment 1):
    // R is multiplied by this after the floor and the confidence inflation. 1.0 = unchanged.
    double r_scale = 1.0;
    double ci_omega_min = 0.3;    // bounds on the CI weight given to the prior
    double ci_omega_max = 0.9;

    // Track score smoothing. nuScenes AMOTA sweeps a threshold over
    // tracking_score to build the recall curve, so handing it the raw
    // per-frame detection score lets a one-frame false positive sit at the
    // same confidence as a long-lived track. The reported score is therefore
    // an EMA of the detection score, scaled by track maturity.
    double score_ema = 0.7;       // weight kept from the previous estimate
};

struct Track {
    int id = -1;
    int cls = -1;
    Eigen::Vector4d x = Eigen::Vector4d::Zero();  // [px, py, vx, vy], LiDAR frame
    Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
    int hits = 0, miss = 0, age = 0;
    double life_s = 0.0;          // seconds since birth (real time, accumulated in predict)
    double coast_s = 0.0;         // seconds since the last matched detection
    double score = 0.0;           // smoothed detection score
    bool confirmed(const Config& c) const { return hits >= c.confirm_hits; }
    // Maturity-weighted score for the tracking submission.
    double report_score(const Config& c) const {
        const double mat = std::min(1.0, (double)hits / std::max(1, c.confirm_hits));
        return score * mat;
    }
    Eigen::Vector2d pos() const { return x.head<2>(); }
    Eigen::Vector2d vel() const { return x.tail<2>(); }
};

class Tracker {
public:
    explicit Tracker(const Config& cfg) : cfg_(cfg) {}

    void reset() {
        tracks_.clear();
        next_id_ = 1;
    }
    const std::vector<Track>& tracks() const { return tracks_; }
    const Config& config() const { return cfg_; }

    // Rotate/translate every track into the current LiDAR frame.
    void ego_compensate(const Eigen::Matrix2d& R_rel, const Eigen::Vector2d& t_rel) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Zero();
        T.block<2, 2>(0, 0) = R_rel;
        T.block<2, 2>(2, 2) = R_rel;  // velocity rotates but does not translate
        for (auto& t : tracks_) {
            const Eigen::Vector2d p = R_rel * t.pos() + t_rel;
            const Eigen::Vector2d v = R_rel * t.vel();
            t.x << p.x(), p.y(), v.x(), v.y();
            t.P = T * t.P * T.transpose();
        }
    }

    void predict(double dt) {
        Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
        A(0, 2) = dt;
        A(1, 3) = dt;
        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        Q(0, 0) = Q(1, 1) = cfg_.q_pos * dt * dt;
        for (auto& t : tracks_) {
            Q(2, 2) = Q(3, 3) = cfg_.q_vel_for(t.cls) * dt;
            t.x = A * t.x;
            t.P = A * t.P * A.transpose() + Q;
            ++t.age;
            t.life_s += dt;
            t.coast_s += dt;   // real time, independent of stepping mode
        }
    }

    // Greedy association on the position Mahalanobis distance.
    // det2trk[i] = index into tracks(), or -1.
    void associate(const std::vector<Eigen::Vector2d>& det_pos,
                   const std::vector<int>& det_cls,
                   std::vector<int>& det2trk) const {
        det2trk.assign(det_pos.size(), -1);
        struct Edge { double c; int ti, dj; };
        std::vector<Edge> edges;
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0) = H(1, 1) = 1;
        for (size_t ti = 0; ti < tracks_.size(); ++ti) {
            const double r = cfg_.r_pos_for(tracks_[ti].cls);
            const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (r * r);
            const Eigen::Matrix2d S = H * tracks_[ti].P * H.transpose() + R;
            const Eigen::Matrix2d Sinv = S.inverse();
            for (size_t dj = 0; dj < det_pos.size(); ++dj) {
                if (cfg_.class_gate && tracks_[ti].cls >= 0 && det_cls[dj] >= 0 &&
                    tracks_[ti].cls != det_cls[dj])
                    continue;
                const Eigen::Vector2d y = det_pos[dj] - tracks_[ti].pos();
                const double m2 = y.transpose() * Sinv * y;
                if (m2 > cfg_.gate_maha2) continue;
                edges.push_back({m2, (int)ti, (int)dj});
            }
        }
        std::sort(edges.begin(), edges.end(),
                  [](const Edge& a, const Edge& b) { return a.c < b.c; });
        std::vector<char> used_t(tracks_.size(), 0), used_d(det_pos.size(), 0);
        for (const auto& e : edges) {
            if (used_t[e.ti] || used_d[e.dj]) continue;
            used_t[e.ti] = used_d[e.dj] = 1;
            det2trk[e.dj] = e.ti;
        }
    }

    void update_position(int ti, const Eigen::Vector2d& z) {
        Track& t = tracks_[ti];
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0) = H(1, 1) = 1;
        const double r = cfg_.r_pos_for(t.cls);
        const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * (r * r);
        const Eigen::Vector2d y = z - H * t.x;
        const Eigen::Matrix2d S = H * t.P * H.transpose() + R;
        const Eigen::Matrix<double, 4, 2> K = t.P * H.transpose() * S.inverse();
        t.x += K * y;
        t.P = (Eigen::Matrix4d::Identity() - K * H) * t.P;
    }

    // Squared Mahalanobis innovation of the last velocity measurement that
    // reached the gate, whether or not it passed. Exposed so the gate can be
    // calibrated against the distribution it actually acts on: the error
    // statistics in the paper's tables are measured against ground truth, but
    // the gate tests innovation against the track prediction, and the two
    // coincide only if that prediction is unbiased with a correct covariance.
    double last_chi2() const { return last_chi2_; }

    // Merge a velocity measurement. Returns false if it was gated out.
    bool update_velocity(int ti, VelMeas m) {
        last_chi2_ = -1.0;
        if (cfg_.fusion == Fusion::None || !m.valid || m.conf < cfg_.conf_min) return false;
        Track& t = tracks_[ti];
        if (cfg_.vel_max_hits > 0 && t.hits > cfg_.vel_max_hits) return false;
        if (cfg_.vel_max_age_s > 0.0 && t.life_s > cfg_.vel_max_age_s + 1e-6) return false;

        // noise floor, then inflate for a low-confidence measurement
        m.R(0, 0) += cfg_.sigma_v_floor * cfg_.sigma_v_floor;
        m.R(1, 1) += cfg_.sigma_v_floor * cfg_.sigma_v_floor;
        const double c = std::clamp(m.conf, 1e-3, 1.0);
        m.R *= (1.0 + 2.0 * (1.0 - c));
        m.R *= cfg_.r_scale;
        if (!m.R.allFinite()) return false;

        const Eigen::Vector2d mu = t.vel();
        const Eigen::Matrix2d Pv = t.P.block<2, 2>(2, 2);

        // innovation gate: a wild measurement must not be allowed to yank the
        // state, which is exactly the failure mode a raw detector spike causes
        const Eigen::Matrix2d S = Pv + m.R;
        const Eigen::Matrix2d Sinv = S.inverse();
        const Eigen::Vector2d y = m.v - mu;
        const double gate = (m.chi2_gate > 0.0) ? m.chi2_gate : cfg_.chi2_gate_vel;
        last_chi2_ = (double)(y.transpose() * Sinv * y);
        if (last_chi2_ > gate) return false;

        if (cfg_.fusion == Fusion::Replace) {
            t.x(2) = m.v.x();
            t.x(3) = m.v.y();
            return true;
        }
        if (cfg_.fusion == Fusion::Kf) {
            Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
            H(0, 2) = H(1, 3) = 1;
            const Eigen::Matrix2d Sf = H * t.P * H.transpose() + m.R;
            const Eigen::Matrix<double, 4, 2> K = t.P * H.transpose() * Sf.inverse();
            t.x += K * (m.v - H * t.x);
            t.P = (Eigen::Matrix4d::Identity() - K * H) * t.P;
            return true;
        }
        // Covariance Intersection. Used when the measurement is not independent
        // of the prior - which is the case in the closed loop, where the DOGM
        // particles were seeded from this track's own velocity. CI stays
        // consistent under unknown correlation and so avoids the double
        // counting that would make the filter overconfident.
        const double omega =
            std::clamp(cfg_.ci_omega_max - (cfg_.ci_omega_max - cfg_.ci_omega_min) * c,
                       cfg_.ci_omega_min, cfg_.ci_omega_max);
        const Eigen::Matrix2d Pinv = Pv.inverse();
        const Eigen::Matrix2d Rinv = m.R.inverse();
        const Eigen::Matrix2d Pn = (omega * Pinv + (1.0 - omega) * Rinv).inverse();
        const Eigen::Vector2d mun = Pn * (omega * Pinv * mu + (1.0 - omega) * Rinv * m.v);
        t.x(2) = mun.x();
        t.x(3) = mun.y();
        t.P.block<2, 2>(2, 2) = Pn;
        return true;
    }

    int spawn(const Eigen::Vector2d& pos, const Eigen::Vector2d& vel, int cls,
              double det_score = 0.0, double p_vel = 4.0) {
        Track t;
        t.id = next_id_++;
        t.cls = cls;
        t.x << pos.x(), pos.y(), vel.x(), vel.y();
        t.P.setIdentity();
        t.P(0, 0) = t.P(1, 1) = 1.0;
        t.P(2, 2) = t.P(3, 3) = p_vel;
        t.hits = 1;
        t.score = det_score;
        tracks_.push_back(t);
        return (int)tracks_.size() - 1;
    }

    // Noise a velocity measurement carries into the filter: floor, confidence inflation and the
    // calibration scale, exactly as update_velocity applies them. Used to initialise a track from
    // a measurement with that measurement's covariance.
    Eigen::Matrix2d measurement_cov(const VelMeas& m) const {
        Eigen::Matrix2d R = m.R;
        R(0, 0) += cfg_.sigma_v_floor * cfg_.sigma_v_floor;
        R(1, 1) += cfg_.sigma_v_floor * cfg_.sigma_v_floor;
        R *= (1.0 + 2.0 * (1.0 - std::clamp(m.conf, 1e-3, 1.0)));
        return R * cfg_.r_scale;
    }
    // Birth with a full velocity covariance (the default spawn uses p_vel * I).
    int spawn_cov(const Eigen::Vector2d& pos, const Eigen::Vector2d& vel, int cls,
                  double det_score, const Eigen::Matrix2d& Pv) {
        const int ti = spawn(pos, vel, cls, det_score);
        tracks_[ti].P.block<2, 2>(2, 2) = Pv;
        return ti;
    }

    void mark_hit(int ti, double det_score) {
        Track& t = tracks_[ti];
        t.hits++;
        t.miss = 0;
        t.coast_s = 0.0;
        t.score = cfg_.score_ema * t.score + (1.0 - cfg_.score_ema) * det_score;
    }
    void mark_miss(int ti) { tracks_[ti].miss++; }

    void cull() {
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const Track& t) {
                                         return t.coast_s > cfg_.max_coast_s;
                                     }),
                      tracks_.end());
    }

private:
    Config cfg_;
    double last_chi2_ = -1.0;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace trk
