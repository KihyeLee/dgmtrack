// dogm_track.cpp - DOGM + tracker with a closed feedback loop.
//
// Data flow per frame:
//
//     detections --> association --> tracks
//                                      |  (track velocity of confirmed tracks)
//                                      v
//                                   DOGM particles           [track -> grid]
//                                      |  (per-box velocity + covariance)
//                                      v
//                            track velocity update           [grid -> track]
//
// The downward arrow is the feedback the project set out to build (Steyer
// et al., see REFERENCES.md [8][9]); `--feedback 0` disables it so the open and
// closed loop can be ablated against each other with nothing else changed.
//
// Why the loop is expected to help the VELOCITY estimate, not just tracking:
// a standalone DOGM has no object identity, so its only cross-frame memory is
// the particle population, and its read-out is a particle-cloud mean whose
// Monte-Carlo noise is redrawn every frame. Association supplies identity,
// which lets the per-object Kalman filter accumulate those read-outs
// analytically instead of resampling them - and the track estimate then seeds
// the particles, which narrows the proposal ([2], informed proposal).
//
// Because the seeded particles are no longer independent of the track prior,
// `--fusion ci` (Covariance Intersection, [23]) is the correct merge in the
// closed loop; a plain Kalman update would double-count the same evidence.
//
// Build:
//   g++ -std=c++17 -O2 -D_USE_MATH_DEFINES src/apps/dogm_track.cpp \
//       -o build/dogm_track.exe -Isrc -Isrc/common -I/c/msys64/ucrt64/include/eigen3

#include <algorithm>
#include <chrono>
#include <sstream>
#include <utility>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "dogm_rfs.hpp"
#include "dogm/grid_motion.hpp"
#include "nusc_io.hpp"
#include "track_kf.hpp"

namespace fs = std::filesystem;
using nusc::json;

static const char* ALGO_VERSION = "dgmtrack-closedloop-3.0.0";

enum class VelSource { None, Detector, Dogm, Fused, Grid, GridFused };

static VelSource parse_source(const std::string& s) {
    if (s == "none") return VelSource::None;
    if (s == "detector") return VelSource::Detector;
    if (s == "fused") return VelSource::Fused;
    if (s == "grid") return VelSource::Grid;
    if (s == "gridfused") return VelSource::GridFused;
    return VelSource::Dogm;
}
static trk::Fusion parse_fusion(const std::string& s) {
    if (s == "none") return trk::Fusion::None;
    if (s == "replace") return trk::Fusion::Replace;
    if (s == "ci") return trk::Fusion::Ci;
    return trk::Fusion::Kf;
}

// Covariance Intersection of two velocity estimates ([23]). Used to merge the
// detector and DOGM measurements before they reach the track, so that neither
// source is assumed independent of the other.
static trk::VelMeas ci_merge(const trk::VelMeas& a, const trk::VelMeas& b) {
    if (!a.valid) return b;
    if (!b.valid) return a;
    const double wa = std::clamp(a.conf / std::max(1e-6, a.conf + b.conf), 0.1, 0.9);
    const Eigen::Matrix2d Ai = a.R.inverse(), Bi = b.R.inverse();
    const Eigen::Matrix2d Pn = (wa * Ai + (1.0 - wa) * Bi).inverse();
    trk::VelMeas m;
    m.v = Pn * (wa * Ai * a.v + (1.0 - wa) * Bi * b.v);
    m.R = Pn;
    m.conf = std::max(a.conf, b.conf);
    // The gate belongs to the detector branch; a merged measurement is still
    // only as trustworthy as its least trustworthy part, so keep the tighter
    // of the two rather than silently dropping it.
    if (a.chi2_gate > 0.0 && b.chi2_gate > 0.0) m.chi2_gate = std::min(a.chi2_gate, b.chi2_gate);
    else m.chi2_gate = std::max(a.chi2_gate, b.chi2_gate);
    m.valid = true;
    return m;
}

// Piecewise-linear lookup of the calibrated velocity noise. The knots are the
// measured per-score-band velocity error of the detector in use; outside the
// measured range the nearest knot is held rather than extrapolated, because an
// extrapolated noise is a guess and a held one is at least a measurement.
static double interp_curve(const std::vector<std::pair<double, double>>& c,
                           double s) {
    if (c.empty()) return 0.5;
    if (s <= c.front().first) return c.front().second;
    if (s >= c.back().first) return c.back().second;
    for (size_t i = 1; i < c.size(); ++i) {
        if (s <= c[i].first) {
            const double t = (s - c[i - 1].first) /
                             std::max(1e-9, c[i].first - c[i - 1].first);
            return c[i - 1].second + t * (c[i].second - c[i - 1].second);
        }
    }
    return c.back().second;
}

// "0.15:0.55,0.45:0.373,0.80:0.253" -> sorted knots
static std::vector<std::pair<double, double>> parse_curve(const std::string& s) {
    std::vector<std::pair<double, double>> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const size_t c = tok.find(':');
        if (c == std::string::npos) continue;
        out.emplace_back(std::stod(tok.substr(0, c)), std::stod(tok.substr(c + 1)));
    }
    std::sort(out.begin(), out.end());
    return out;
}

// "car:0.3,pedestrian:1.2" -> per-class vector indexed by nusc::class_id; classes
// not named keep the global value.
static std::vector<double> parse_class_values(const std::string& spec, double fallback) {
    std::vector<double> out(nusc::tracking_classes().size(), fallback);
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const size_t c = tok.find(':');
        if (c == std::string::npos) continue;
        const int id = nusc::class_id(tok.substr(0, c));
        if (id < 0) throw std::runtime_error("unknown class in class value list: " + tok);
        out[id] = std::stod(tok.substr(c + 1));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string pose_cache, det_dir, out_dir, dataroot, scenes_arg, lidar_cache;
    std::string src_name = "dogm", fusion_name = "kf", report = "track";
    // Which position is submitted for a matched detection. "det" reports the
    // detector box unchanged, so no velocity source can move a box and a
    // localisation metric can only react to identities, scores and the
    // evaluator's interpolation. "state" reports the track's posterior
    // position after this frame's update; size, heading and z stay from the
    // detection. Unmatched detections are never submitted either way.
    std::string box_output = "det";          // det | state
    double score_thr = 0.0, det_vel_sigma = 0.5;
    // Growth rate of the detector velocity noise as the detection score falls.
    // 0 = fixed noise (the configuration used for every result before this).
    double det_kappa = 0.0;
    // Calibrated alternative to det_kappa: measured "score:sigma" knots.
    std::string det_sigma_curve_arg;
    std::vector<std::pair<double, double>> det_sigma_curve;
    // Calibrated robust gate: measured "score:P(gross error)" knots.
    std::string det_gate_curve_arg;
    std::vector<std::pair<double, double>> det_gate_curve;
    std::string innov_log_path;
    bool feedback = true;
    // Whether the tracker is updated on every 20 Hz sweep or only on the 2 Hz
    // keyframes. This is not a tuning knob - it decides whether velocity is
    // load-bearing at all. Stepping on sweeps means a position measurement
    // arrives every 0.05 s, so the constant-velocity prediction barely
    // matters and every velocity source scores the same (measured: car AMOTA
    // 0.785 for none/detector/dogm/fused alike). Keyframe stepping has to
    // extrapolate 0.5 s, which is where velocity quality actually shows up -
    // and is what the legacy tracker did.
    bool update_on_sweeps = false;
    // Coasting output. A track with no detection this keyframe can still be
    // reported at its predicted pose. "blind" always reports it; "dogm" only
    // reports it when the occupancy grid still shows returns inside the
    // predicted footprint. The pair isolates what the DOGM contributes: blind
    // coasting alone already raises recall, so only dogm-over-blind is
    // evidence that the grid carries information the tracker did not have.
    std::string coast_output = "none";       // none | blind | dogm
    double coast_support_min = 0.15;         // occupied fraction required
    double coast_output_max_s = 1.5;         // stop reporting after this long
    dogm::Config dcfg;
    gm::Config gcfg;
    trk::Config tcfg;
    double grid_vel_sigma = 0.8;   // m/s, base noise of the grid cue
    bool grid_diag = false;        // dump per-estimate records
    // class rules for the grid velocity (rule A / rule B, PREREGISTRATION 2026-09-24)
    std::set<std::string> grid_exclude;          // --grid_exclude_classes a,b
    std::map<std::string, double> grid_prior;    // --grid_class_prior cls:sigma,...  (m/s)
    std::string grid_exclude_arg, grid_prior_arg;
    // per-frame stage timing (off unless --timing_log is given; outputs are unaffected)
    std::string timing_log_path;
    // Initialise a track born from a valid grid velocity with that measurement's covariance
    // instead of the fixed p_vel (PREREGISTRATION 2026-09-24, experiment 1). Off by default.
    bool birth_cov_from_meas = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--pose_cache") pose_cache = val();
        else if (a == "--det_dir") det_dir = val();
        else if (a == "--out_dir") out_dir = val();
        else if (a == "--dataroot") dataroot = val();
        else if (a == "--lidar_cache") lidar_cache = val();
        else if (a == "--scenes") scenes_arg = val();
        else if (a == "--score_thr") score_thr = std::stod(val());
        else if (a == "--vel_source") src_name = val();
        else if (a == "--fusion") fusion_name = val();
        else if (a == "--report") report = val();           // track | dogm
        else if (a == "--box_output") box_output = val();   // det | state
        else if (a == "--feedback") feedback = (val() != "0");
        else if (a == "--det_vel_sigma") det_vel_sigma = std::stod(val());
        else if (a == "--det_kappa") det_kappa = std::stod(val());
        else if (a == "--det_sigma_curve") det_sigma_curve_arg = val();
        else if (a == "--det_gate_curve") det_gate_curve_arg = val();
        else if (a == "--innov_log") innov_log_path = val();
        else if (a == "--seed") dcfg.seed = (uint64_t)std::stoull(val());
        else if (a == "--particles_per_box") dcfg.particles_per_box = std::stoi(val());
        else if (a == "--dogm_fresh_count") dcfg.fresh_birth_count = std::stoi(val()) != 0;
        else if (a == "--dogm_budget") dcfg.resample_budget = std::stoi(val()) != 0;
        else if (a == "--sigma_v_proc") dcfg.sigma_v_proc = std::stod(val());
        else if (a == "--birth_seed_fraction") dcfg.birth_seed_fraction = std::stod(val());
        else if (a == "--birth_static_fraction") dcfg.birth_static_fraction = std::stod(val());
        else if (a == "--vel_std_abs") dcfg.vel_std_abs = std::stod(val());
        else if (a == "--vel_std_rel") dcfg.vel_std_rel = std::stod(val());
        else if (a == "--q_vel") tcfg.q_vel = std::stod(val());
        else if (a == "--gate_maha2") tcfg.gate_maha2 = std::stod(val());
        else if (a == "--vel_max_hits") tcfg.vel_max_hits = std::stoi(val());
        else if (a == "--vel_conf_min") tcfg.conf_min = std::stod(val());
        else if (a == "--seed_conf_min") tcfg.seed_conf_min = std::stod(val());
        else if (a == "--class_q_vel") tcfg.q_vel_by_cls = parse_class_values(val(), tcfg.q_vel);
        else if (a == "--class_r_pos") tcfg.r_pos_by_cls = parse_class_values(val(), tcfg.r_pos);
        else if (a == "--q_pos") tcfg.q_pos = std::stod(val());
        else if (a == "--max_coast_s") tcfg.max_coast_s = std::stod(val());
        else if (a == "--update_on_sweeps") update_on_sweeps = (val() != "0");
        else if (a == "--coast_output") coast_output = val();
        else if (a == "--grid_res") gcfg.resolution = std::stod(val());
        else if (a == "--grid_search_m") gcfg.search_radius_m = std::stod(val());
        else if (a == "--grid_min_peak") gcfg.min_peak_score = std::stod(val());
        else if (a == "--grid_vel_sigma") grid_vel_sigma = std::stod(val());
        else if (a == "--grid_min_cells") gcfg.min_cells = std::stoi(val());
        else if (a == "--grid_diag") grid_diag = true;
        else if (a == "--grid_exclude_classes") grid_exclude_arg = val();
        else if (a == "--grid_class_prior") grid_prior_arg = val();
        else if (a == "--timing_log") timing_log_path = val();
        else if (a == "--grid_r_scale") tcfg.r_scale = std::stod(val());
        else if (a == "--vel_max_age_s") tcfg.vel_max_age_s = std::stod(val());
        else if (a == "--birth_cov_from_meas") birth_cov_from_meas = (val() != "0");
        else if (a == "--coast_support_min") coast_support_min = std::stod(val());
        else if (a == "--coast_output_max_s") coast_output_max_s = std::stod(val());
        else if (a == "--confirm_hits") tcfg.confirm_hits = std::stoi(val());
        else if (a == "--chi2_gate_vel") tcfg.chi2_gate_vel = std::stod(val());
        else if (a == "--help") {
            std::cerr <<
                "usage: dogm_track --pose_cache <json> --det_dir <dir> --out_dir <dir>\n"
                "                  --dataroot <nuscenes root> [options]\n"
                "  --vel_source none|detector|dogm|fused   velocity fed to the track\n"
                "  --fusion     none|replace|kf|ci         how it is merged\n"
                "  --feedback   0|1                        track -> grid seeding\n"
                "  --report     track|dogm                 which velocity is written out\n"
                "  --box_output det|state                  submitted position of a matched box\n";
            return 0;
        }
    }
    {
        std::stringstream ss(grid_exclude_arg);
        std::string s;
        while (std::getline(ss, s, ',')) if (!s.empty()) grid_exclude.insert(s);
        std::stringstream sp(grid_prior_arg);
        while (std::getline(sp, s, ',')) {
            const auto c = s.find(':');
            if (c != std::string::npos) grid_prior[s.substr(0, c)] = std::stod(s.substr(c + 1));
        }
    }
    if (pose_cache.empty() || det_dir.empty() || out_dir.empty() || dataroot.empty()) {
        std::cerr << "ERROR: --pose_cache, --det_dir, --out_dir, --dataroot required\n";
        return 1;
    }
    const VelSource vsrc = parse_source(src_name);
    tcfg.fusion = parse_fusion(fusion_name);
    dcfg.grid.finalize();
    if (!feedback) dcfg.birth_seed_fraction = 0.0;  // open loop: never seed from tracks

    std::unordered_map<std::string, nusc::PoseEntry> poses;
    if (!nusc::load_pose_cache(pose_cache, poses)) {
        std::cerr << "cannot read pose cache " << pose_cache << "\n";
        return 1;
    }
    std::cerr << "[dogm_track] pose cache: " << poses.size() << " frames\n";

    std::vector<std::string> scene_files;
    det_sigma_curve = parse_curve(det_sigma_curve_arg);
    det_gate_curve = parse_curve(det_gate_curve_arg);

    std::ofstream innov_log;
    if (!innov_log_path.empty()) {
        fs::create_directories(fs::path(innov_log_path).parent_path());
        innov_log.open(innov_log_path);
    }

    if (!scenes_arg.empty()) {
        std::stringstream ss(scenes_arg);
        std::string s;
        while (std::getline(ss, s, ','))
            if (!s.empty()) scene_files.push_back(det_dir + "/" + s + ".jsonl");
    } else {
        for (const auto& e : fs::directory_iterator(det_dir))
            if (e.is_regular_file() && e.path().extension() == ".jsonl")
                scene_files.push_back(e.path().string());
        std::sort(scene_files.begin(), scene_files.end());
    }
    fs::create_directories(out_dir);
    std::ofstream gdiag;
    std::ofstream tlog;
    if (!timing_log_path.empty()) tlog.open(timing_log_path, std::ios::trunc);
    using Clk = std::chrono::steady_clock;
    auto msd = [](Clk::time_point a, Clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    if (grid_diag) gdiag.open(out_dir + "/grid_diag.jsonl", std::ios::trunc);

    dogm::DogmRfs filter(dcfg);
    gm::GridMotion gmotion(gcfg);
    trk::Tracker tracker(tcfg);
    // last matched shape per track id, needed to emit a box while coasting
    struct TrackShape { std::vector<double> size, rot; std::string name; double yaw_l, w, l; };
    std::unordered_map<int, TrackShape> shape;
    // nuScenes tracking submission format: sample_token -> [box, ...]
    json track_results = json::object();
    size_t total_frames = 0, n_dogm_used = 0, n_det_used = 0, n_out_boxes = 0,
           n_coast_boxes = 0;
    const auto t0 = std::chrono::steady_clock::now();

    for (const auto& sf : scene_files) {
        const std::string scene = fs::path(sf).stem().string();
        auto frames = nusc::load_scene(sf);
        if (frames.empty()) continue;

        filter.reset();
        gmotion.reset();
        tracker.reset();
        shape.clear();
        bool have_prev = false;
        double key_t = -1.0;   // time of the last keyframe, for accumulation
        Eigen::Matrix3d R_prev = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t_prev = Eigen::Vector3d::Zero();
        int64_t prev_ts = -1;

        std::ofstream out(out_dir + "/" + scene + ".jsonl", std::ios::trunc);

        for (const auto& fr : frames) {
            auto pit = poses.find(fr.sd_token);
            if (pit == poses.end()) continue;
            const nusc::PoseEntry& pe = pit->second;
            const auto tm0 = Clk::now();

            Eigen::Matrix3d R_cur;
            Eigen::Vector3d t_cur;
            nusc::lidar_pose(pe, R_cur, t_cur);

            Eigen::Matrix2d R_rel = Eigen::Matrix2d::Identity();
            Eigen::Vector2d t_rel = Eigen::Vector2d::Zero();
            if (have_prev) nusc::relative_2d(R_cur, t_cur, R_prev, t_prev, R_rel, t_rel);
            double dt = 0.05;
            if (prev_ts > 0) dt = std::clamp((fr.ts - prev_ts) / 1e6, 0.01, 0.5);

            // ---- detections into the LiDAR frame
            std::vector<nusc::BoxInLidar> boxes;
            for (size_t i = 0; i < fr.boxes.size(); ++i) {
                const auto& d = fr.boxes[i];
                if (nusc::class_id(d.name) < 0 || d.score < score_thr) continue;
                auto b = nusc::box_to_lidar(d, pe, (int)i);
                if (b.cx * b.cx + b.cy * b.cy <= 4.0) continue;  // ego self-return
                boxes.push_back(b);
            }

            // ---- tracker prediction in the current frame
            tracker.ego_compensate(R_rel, t_rel);
            tracker.predict(dt);

            std::vector<Eigen::Vector2d> det_pos(boxes.size());
            std::vector<int> det_cls(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i) {
                det_pos[i] = Eigen::Vector2d(boxes[i].cx, boxes[i].cy);
                det_cls[i] = boxes[i].cls;
            }
            std::vector<int> det2trk;
            tracker.associate(det_pos, det_cls, det2trk);

            // ---- [track -> grid] seed velocities from confirmed tracks
            std::vector<Eigen::Vector2d> seed_vel;
            if (feedback) {
                seed_vel.assign(boxes.size(), Eigen::Vector2d::Zero());
                for (size_t i = 0; i < boxes.size(); ++i) {
                    const int ti = det2trk[i];
                    if (ti < 0) continue;
                    const auto& t = tracker.tracks()[ti];
                    // Only confirmed tracks steer the grid: an immature track is
                    // itself uncertain, and letting it seed particles would feed
                    // its own error back and lock the loop onto it.
                    if (t.confirmed(tracker.config())) seed_vel[i] = t.vel();
                }
            }

            // ---- DOGM cycle
            std::vector<dogm::BoxLidar2D> dboxes(boxes.size());
            for (size_t i = 0; i < boxes.size(); ++i)
                dboxes[i] = {boxes[i].cx, boxes[i].cy, boxes[i].w, boxes[i].l, boxes[i].yaw};
            std::vector<Eigen::Vector2f> pts;
            const auto tm1 = Clk::now();
            if (!nusc::load_lidar_xy_cached(lidar_cache, pe.scene, fr.sd_token, pts))
                pts = nusc::load_lidar_xy(dataroot + "/" + pe.file);
            const auto tm2 = Clk::now();
            std::vector<dogm::BoxVelocity> dvel;
            filter.step(R_rel, t_rel, dt, pts, dboxes, seed_vel, dvel);
            const auto tm3 = Clk::now();

            // ---- grid motion cue.
            // Every sweep contributes its in-box returns to a per-track
            // accumulator (ego-compensated by working in the global frame and
            // motion-compensated by the track's own velocity); the correlation
            // itself runs once per keyframe, over the 0.5 s baseline.
            std::vector<gm::Estimate> gest(boxes.size());
            const bool use_grid =
                (vsrc == VelSource::Grid || vsrc == VelSource::GridFused);
            if (use_grid) {
                for (size_t i = 0; i < boxes.size(); ++i) {
                    const int ti = det2trk[i];
                    if (ti < 0) continue;
                    const int tid_i = tracker.tracks()[ti].id;
                    dogm::BoxLidar2D qb{boxes[i].cx, boxes[i].cy,
                                        boxes[i].w, boxes[i].l, boxes[i].yaw};
                    std::vector<Eigen::Vector2d> inbox;
                    inbox.reserve(512);
                    for (const auto& q : pts) {
                        if (!dogm::DogmRfs::point_in_box(q.x(), q.y(), qb)) continue;
                        const Eigen::Vector3d gpt =
                            nusc::pos_to_global(pe, q.x(), q.y(), 0.0);
                        inbox.emplace_back(gpt.x(), gpt.y());
                    }
                    const Eigen::Vector2d vg_track =
                        nusc::vel_to_global(pe, tracker.tracks()[ti].vel());
                    const double t_s = fr.ts * 1e-6;
                    gmotion.accumulate(tid_i, key_t > 0 ? key_t : t_s, t_s,
                                       inbox, vg_track);
                    if (fr.is_key) {
                        gest[i] = gmotion.estimate_accum(tid_i, t_s, vg_track * 0.5);
                        if (grid_diag && gdiag) {
                            const Eigen::Vector2d dv = nusc::vel_to_global(
                                pe, Eigen::Vector2d(boxes[i].vx, boxes[i].vy));
                            gdiag << "{\"n_cells\":" << gest[i].n_cells
                                  << ",\"peak\":" << gest[i].peak
                                  << ",\"conf\":" << gest[i].conf
                                  << ",\"valid\":" << (gest[i].valid ? 1 : 0)
                                  << ",\"v_grid\":[" << gest[i].v.x() << ","
                                  << gest[i].v.y() << "],\"v_det\":[" << dv.x()
                                  << "," << dv.y() << "]}" << std::endl;
                        }
                    }
                }
                if (fr.is_key) key_t = fr.ts * 1e-6;
            }

            const auto tm4 = Clk::now();
            // ---- [grid -> track] velocity update, then births.
            // On sweeps in keyframe-stepping mode the DOGM still runs (it
            // needs the sweeps to accumulate evidence) and the tracks are
            // still predicted, but nothing is matched, spawned or aged.
            const bool do_update = fr.is_key || update_on_sweeps;
            for (size_t i = 0; do_update && i < boxes.size(); ++i) {
                trk::VelMeas m_dogm, m_det;
                if (dvel[i].valid) {
                    m_dogm.v = dvel[i].v;
                    m_dogm.R = dvel[i].S;
                    m_dogm.conf = dvel[i].conf;
                    m_dogm.valid = true;
                }
                if (boxes[i].has_vel) {
                    m_det.v = Eigen::Vector2d(boxes[i].vx, boxes[i].vy);
                    // Confidence-adaptive measurement noise.
                    //
                    // A fixed sigma (det_kappa = 0) trusts the detector
                    // velocity equally at score 0.9 and at score 0.15. Measured
                    // on the dev set, that is wrong by more than a factor of
                    // two:
                    //
                    //   score >= 0.6   EPE 0.253  jitter 0.471  spike>6 0.0035
                    //   score 0.3-0.6  EPE 0.373  jitter 0.583  spike>6 0.0118
                    //   score <  0.3   EPE 0.550  jitter 0.679  spike>6 0.0289
                    //
                    // The grid estimate does not degrade the same way. Below
                    // 0.3 it is both more accurate (0.498) and far steadier
                    // (jitter 0.362, spike 0.0037), because it is a recursive
                    // estimate rather than a per-frame regression. So the noise
                    // the filter assigns to the detector should grow as the
                    // detection gets less certain, and the fusion then leans on
                    // the grid exactly where the grid is the better source.
                    //
                    //   sigma_det(s) = sigma_0 * (1 + kappa * (1 - s))
                    //
                    // kappa = 2.8 reproduces the measured 2.17x EPE ratio
                    // between the <0.3 and >=0.6 bands at their mean scores.
                    // kappa = 0 recovers the previous fixed model exactly, so
                    // the old configuration is nested inside this one and the
                    // ablation is clean.
                    //
                    // CALIBRATED FORM (--det_sigma_curve). Picking kappa by
                    // hand, or by eye from a table, is the part of this that is
                    // not reproducible across detectors: the rate at which a
                    // detector's velocity degrades with its own score is a
                    // property of that detector, not a constant of nature. The
                    // curve option therefore takes the measured error directly,
                    // as "score:sigma" pairs from a held-out split, and
                    // interpolates between them. Then nothing is tuned - the
                    // noise model is a measurement, and the same one-command
                    // procedure (tools/eval_velocity.py, stratified by score)
                    // produces it for any detector.
                    const double s_det = fr.boxes[boxes[i].orig].score;
                    double sd;
                    if (!det_sigma_curve.empty()) {
                        sd = interp_curve(det_sigma_curve, s_det);
                    } else {
                        sd = det_vel_sigma * (1.0 + det_kappa * (1.0 - s_det));
                    }
                    m_det.R = Eigen::Matrix2d::Identity() * (sd * sd);
                    // conf drives the CI weighting; hold it at 1.0 when neither
                    // adaptive model is on so the baseline is bit-identical.
                    m_det.conf = (det_kappa > 0.0 || !det_sigma_curve.empty())
                                     ? std::clamp(s_det, 0.05, 1.0)
                                     : 1.0;
                    // Calibrated robust gate. The knots hold the MEASURED
                    // probability that this detector's velocity is grossly
                    // wrong at this score; the gate is the chi-square quantile
                    // that rejects exactly that often. With two degrees of
                    // freedom the survival function is exp(-x/2), so
                    //      chi2(p) = -2 ln p.
                    // At the measured rates that is 5.8 for the lowest score
                    // band (p = 0.055) and 10.8 for the highest (p = 0.0046):
                    // the filter disbelieves a surprising velocity from an
                    // unsure detector sooner than from a confident one, by
                    // exactly the margin the detector's own error statistics
                    // justify. Nothing here is hand-set.
                    if (!det_gate_curve.empty()) {
                        const double p = std::clamp(
                            interp_curve(det_gate_curve, s_det), 1e-4, 0.5);
                        m_det.chi2_gate = -2.0 * std::log(p);
                    }
                    m_det.valid = true;
                }
                trk::VelMeas m_grid;
                if (gest[i].valid) {
                    // the cue is produced in the global frame; the tracker
                    // works in the LiDAR frame
                    const Eigen::Vector3d vg(gest[i].v.x(), gest[i].v.y(), 0.0);
                    const Eigen::Vector3d vl =
                        pe.cs_q.inverse() * (pe.ep_q.inverse() * vg);
                    m_grid.v = Eigen::Vector2d(vl.x(), vl.y());
                    const double sg = grid_vel_sigma /
                                      std::max(0.2, gest[i].conf);
                    m_grid.R = Eigen::Matrix2d::Identity() * (sg * sg);
                    m_grid.conf = gest[i].conf;
                    m_grid.valid = true;
                }
                trk::VelMeas m;
                switch (vsrc) {
                    case VelSource::None: break;
                    case VelSource::Detector: m = m_det; break;
                    case VelSource::Dogm: m = m_dogm; break;
                    case VelSource::Fused: m = ci_merge(m_det, m_dogm); break;
                    case VelSource::Grid: m = m_grid; break;
                    case VelSource::GridFused: m = ci_merge(m_det, m_grid); break;
                }
                // Class rules for the grid velocity (PREREGISTRATION 2026-09-24). Both
                // default off, so every earlier configuration is unchanged. The
                // measurement m feeds the early update and the seed of a new track,
                // so one edit here covers both places the policy uses the grid.
                //   exclude: the class never uses the grid velocity (rule A).
                //   prior:   the grid velocity is shrunk toward zero by the class's
                //            velocity prior N(0, s_c^2 I) from the training split,
                //            v <- s^2 (s^2 I + R)^-1 v with R the same inflated noise
                //            update_velocity applies (rule B).
                if (m.valid && vsrc != VelSource::Detector && vsrc != VelSource::None) {
                    const std::string& cname = fr.boxes[boxes[i].orig].name;
                    if (grid_exclude.count(cname)) {
                        m.valid = false;
                    } else if (grid_prior.count(cname)) {
                        const double s2 = grid_prior.at(cname) * grid_prior.at(cname);
                        const double fl = tracker.config().sigma_v_floor;
                        Eigen::Matrix2d R = m.R + Eigen::Matrix2d::Identity() * (fl * fl);
                        R *= 1.0 + 2.0 * (1.0 - std::clamp(m.conf, 1e-3, 1.0));
                        const Eigen::Matrix2d K =
                            s2 * (s2 * Eigen::Matrix2d::Identity() + R).inverse();
                        m.v = K * m.v;
                    }
                }

                const int ti = det2trk[i];
                if (ti >= 0) {
                    tracker.update_position(ti, det_pos[i]);
                    const bool vel_ok = tracker.update_velocity(ti, m);
                    // Innovation log. The gate acts on the innovation against
                    // the track prediction, but the error statistics used to
                    // set it are measured against ground truth. Recording the
                    // innovation alongside the detection score and the reported
                    // velocity lets the threshold be calibrated on the quantity
                    // it actually tests, instead of on a proxy for it.
                    if (innov_log.is_open() && tracker.last_chi2() >= 0.0) {
                        innov_log << "{\"scene\":\"" << scene
                                  << "\",\"sample\":\"" << fr.sample_token
                                  << "\",\"t\":"
                                  << (fr.ts * 1e-6) << ",\"score\":"
                                  << fr.boxes[boxes[i].orig].score
                                  << ",\"chi2\":" << tracker.last_chi2()
                                  << ",\"cls\":\"" << fr.boxes[boxes[i].orig].name
                                  << "\",\"dvx\":" << fr.boxes[boxes[i].orig].vx
                                  << ",\"dvy\":" << fr.boxes[boxes[i].orig].vy
                                  << ",\"gx\":" << fr.boxes[boxes[i].orig].t[0]
                                  << ",\"gy\":" << fr.boxes[boxes[i].orig].t[1]
                                  << ",\"passed\":" << (vel_ok ? 1 : 0) << "}"
                                  << std::endl;
                    }
                    if (vel_ok) {
                        if (vsrc == VelSource::Dogm) ++n_dogm_used;
                        else if (vsrc == VelSource::Detector) ++n_det_used;
                    }
                    tracker.mark_hit(ti, fr.boxes[boxes[i].orig].score);
                    {
                        const auto& d = fr.boxes[boxes[i].orig];
                        shape[tracker.tracks()[ti].id] =
                            {d.s, d.q, d.name, boxes[i].yaw, boxes[i].w, boxes[i].l};
                    }
                } else {
                    // Legacy seeding by default; --seed_conf_min adds a birth gate.
                    const bool seed_ok = m.valid &&
                        (tracker.config().seed_conf_min < 0.0 ||
                         m.conf >= tracker.config().seed_conf_min);
                    const Eigen::Vector2d v0 = seed_ok ? m.v : Eigen::Vector2d::Zero();
                    if (seed_ok && birth_cov_from_meas)
                        det2trk[i] = tracker.spawn_cov(det_pos[i], v0, det_cls[i],
                                                       fr.boxes[boxes[i].orig].score,
                                                       tracker.measurement_cov(m));
                    else
                        det2trk[i] = tracker.spawn(det_pos[i], v0, det_cls[i],
                                                   fr.boxes[boxes[i].orig].score);
                }
            }
            if (do_update) {
                for (size_t ti = 0; ti < tracker.tracks().size(); ++ti) {
                    bool matched = false;
                    for (int d : det2trk)
                        if (d == (int)ti) { matched = true; break; }
                    if (!matched) tracker.mark_miss((int)ti);
                }
            }

            have_prev = true;
            R_prev = R_cur;
            t_prev = t_cur;
            prev_ts = fr.ts;
            ++total_frames;

            const auto tm5 = Clk::now();
            // ---- output (keyframes only; that is what the evaluation reads)
            if (fr.is_key) {
                std::vector<std::array<double, 2>> vg(fr.boxes.size(), {0.0, 0.0});
                std::vector<char> valid(fr.boxes.size(), 0);
                std::vector<double> conf(fr.boxes.size(), 0.0);
                std::vector<int> tid(fr.boxes.size(), -1);
                std::vector<char> tconf(fr.boxes.size(), 0);   // track confirmed
                std::vector<double> tscore(fr.boxes.size(), 0.0);
                // Posterior track position in global XY for matched boxes.
                std::vector<std::array<double, 2>> tpos(fr.boxes.size(), {0.0, 0.0});
                // Grid velocity covariance in the global frame, [Sxx, Sxy, Syy], and
                // its support, so an external tracker can use the same measurement.
                std::vector<std::array<double, 3>> gcov(fr.boxes.size(), {0.0, 0.0, 0.0});
                std::vector<std::array<int, 2>> gsup(fr.boxes.size(), {0, 0});
                for (size_t i = 0; i < boxes.size(); ++i) {
                    const int o = boxes[i].orig;
                    Eigen::Vector2d v_l = Eigen::Vector2d::Zero();
                    bool ok = false;
                    if (report == "dogm") {
                        if (dvel[i].valid) {
                            v_l = dvel[i].v; ok = true; conf[o] = dvel[i].conf;
                            const Eigen::Matrix3d Rg =
                                (pe.ep_q * pe.cs_q).toRotationMatrix();
                            const Eigen::Matrix2d R2 = Rg.block<2, 2>(0, 0);
                            const Eigen::Matrix2d Sg = R2 * dvel[i].S * R2.transpose();
                            gcov[o] = {Sg(0, 0), Sg(0, 1), Sg(1, 1)};
                            gsup[o] = {dvel[i].n_cells, dvel[i].n_particles};
                        }
                    } else {
                        const int ti = det2trk[i];
                        if (ti >= 0) {
                            const auto& t = tracker.tracks()[ti];
                            v_l = t.vel();
                            ok = t.confirmed(tracker.config());
                            conf[o] = dvel[i].valid ? dvel[i].conf : 0.0;
                            tid[o] = t.id;
                            tconf[o] = ok ? 1 : 0;
                            tscore[o] = t.report_score(tracker.config());
                            const Eigen::Vector3d gp =
                                nusc::pos_to_global(pe, t.x(0), t.x(1), 0.0);
                            tpos[o] = {gp.x(), gp.y()};
                        }
                    }
                    if (ok) {
                        const Eigen::Vector2d g = nusc::vel_to_global(pe, v_l);
                        vg[o] = {g.x(), g.y()};
                        valid[o] = 1;
                    }
                }

                json line;
                line["sample_token"] = fr.sample_token;
                line["sd_token"] = fr.sd_token;
                line["is_key_frame"] = fr.is_key;
                line["timestamp"] = fr.ts;
                json arr = json::array();
                for (size_t i = 0; i < fr.boxes.size(); ++i) {
                    const auto& d = fr.boxes[i];
                    arr.push_back({{"translation", d.t},
                                   {"size", d.s},
                                   {"rotation", d.q},
                                   {"velocity", {vg[i][0], vg[i][1]}},
                                   {"detection_name", d.name},
                                   {"detection_score", d.score},
                                   {"attribute_name", d.attr},
                                   {"dogm_valid", (bool)valid[i]},
                                   {"dogm_conf", conf[i]},
                                   {"tracking_id", tid[i]},
                                   {"track_translation",
                                    {tpos[i][0], tpos[i][1], d.t[2]}},
                                   {"dogm_cov", {gcov[i][0], gcov[i][1], gcov[i][2]}},
                                   {"dogm_cells", gsup[i][0]},
                                   {"dogm_particles", gsup[i][1]}});
                }
                line["boxes"] = arr;
                out << line.dump() << "\n";

                // nuScenes tracking submission format. Track ids are prefixed
                // with the scene so they stay unique once every scene is
                // merged into a single submission file.
                json tarr = json::array();
                for (size_t i = 0; i < boxes.size(); ++i) {
                    const int o = boxes[i].orig;
                    // Everything with a track id is submitted. AMOTA sweeps a
                    // threshold over tracking_score, so withholding immature
                    // tracks does not remove them from the curve - it removes
                    // the high-recall end of the curve permanently (measured:
                    // recall 0.742 -> 0.639 when they were gated out).
                    // Ranking, not filtering, is what the metric rewards, so
                    // maturity is expressed through tracking_score instead.
                    if (tid[o] < 0) continue;
                    const auto& d = fr.boxes[o];
                    const std::vector<double> submitted_t =
                        (box_output == "state")
                            ? std::vector<double>{tpos[o][0], tpos[o][1], d.t[2]}
                            : d.t;
                    tarr.push_back(
                        {{"sample_token", fr.sample_token},
                         {"translation", submitted_t},
                         {"size", d.s},
                         {"rotation", d.q},
                         {"velocity", {vg[o][0], vg[o][1]}},
                         {"tracking_id", scene + "_" + std::to_string(tid[o])},
                         {"tracking_name", d.name},
                         {"tracking_score", tscore[o]}});
                    ++n_out_boxes;
                }
                // Tracks with no detection this keyframe, reported at their
                // predicted pose (see --coast_output).
                if (coast_output != "none") {
                    std::vector<char> emitted(tracker.tracks().size(), 0);
                    for (size_t i = 0; i < boxes.size(); ++i)
                        if (det2trk[i] >= 0) emitted[det2trk[i]] = 1;
                    for (size_t ti = 0; ti < tracker.tracks().size(); ++ti) {
                        const auto& t = tracker.tracks()[ti];
                        if (emitted[ti] || t.coast_s <= 0.0) continue;
                        if (t.coast_s > coast_output_max_s) continue;
                        if (!t.confirmed(tracker.config())) continue;
                        auto sh = shape.find(t.id);
                        if (sh == shape.end()) continue;

                        dogm::BoxLidar2D q;
                        q.cx = t.x(0); q.cy = t.x(1);
                        q.w = sh->second.w; q.l = sh->second.l; q.yaw = sh->second.yaw_l;
                        double support = -1.0;
                        if (coast_output == "dogm") {
                            support = filter.box_support(q);
                            if (support < coast_support_min) continue;
                        }
                        const Eigen::Vector3d gp =
                            nusc::pos_to_global(pe, t.x(0), t.x(1), 0.0);
                        const Eigen::Vector2d gv = nusc::vel_to_global(pe, t.vel());
                        // keep the z of the last matched detection
                        tarr.push_back(
                            {{"sample_token", fr.sample_token},
                             {"translation", {gp.x(), gp.y(), 0.0}},
                             {"size", sh->second.size},
                             {"rotation", sh->second.rot},
                             {"velocity", {gv.x(), gv.y()}},
                             {"tracking_id", scene + "_" + std::to_string(t.id)},
                             {"tracking_name", sh->second.name},
                             {"tracking_score", t.report_score(tracker.config())}});
                        ++n_coast_boxes;
                    }
                }
                track_results[fr.sample_token] = std::move(tarr);
            }

            if (do_update) tracker.cull();
            if (tlog) {
                const auto tm6 = Clk::now();
                tlog << "{\"scene\":\"" << scene << "\",\"key\":" << (fr.is_key ? 1 : 0)
                     << ",\"boxes\":" << boxes.size() << ",\"points\":" << pts.size()
                     << ",\"particles\":" << filter.particle_count()
                     << ",\"ms_tracker_pre\":" << msd(tm0, tm1) << ",\"ms_lidar_read\":" << msd(tm1, tm2)
                     << ",\"ms_dogm\":" << msd(tm2, tm3) << ",\"ms_grid_motion\":" << msd(tm3, tm4)
                     << ",\"ms_tracker_post\":" << msd(tm4, tm5) << ",\"ms_output\":" << msd(tm5, tm6)
                     << ",\"ms_total\":" << msd(tm0, tm6) << "}\n";
            }
        }
        std::cerr << "  " << scene << " (" << frames.size() << " frames)\n";
    }

    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    std::cerr << "[dogm_track] " << total_frames << " frames in " << secs << "s ("
              << (total_frames ? secs / total_frames * 1000.0 : 0.0) << " ms/frame)\n";

    {
        json sub;
        sub["meta"] = {{"use_camera", false}, {"use_lidar", true},
                       {"use_radar", false},  {"use_map", false},
                       {"use_external", false}, {"algo_version", ALGO_VERSION},
                       {"vel_source", src_name},
        {"grid", {{"resolution", gcfg.resolution},
                  {"search_radius_m", gcfg.search_radius_m},
                  {"min_peak_score", gcfg.min_peak_score},
                  {"dt_min", gcfg.dt_min}, {"dt_max", gcfg.dt_max},
                  {"vel_sigma", grid_vel_sigma}}}, {"fusion", fusion_name},
                       {"feedback", feedback}};
        sub["results"] = track_results;
        std::ofstream tf(out_dir + "/tracking_results.json");
        tf << sub.dump() << std::endl;
        std::cerr << "[dogm_track] wrote tracking_results.json ("
                  << track_results.size() << " samples, " << n_out_boxes
                  << " boxes)" << std::endl;
    }

    json meta = {
        {"algo_version", ALGO_VERSION},
        {"vel_source", src_name},
        {"grid", {{"resolution", gcfg.resolution},
                  {"search_radius_m", gcfg.search_radius_m},
                  {"min_peak_score", gcfg.min_peak_score},
                  {"dt_min", gcfg.dt_min}, {"dt_max", gcfg.dt_max},
                  {"vel_sigma", grid_vel_sigma}}},
        {"fusion", fusion_name},
        {"feedback", feedback},
        {"report", report},
        {"box_output", box_output},
        {"seed", dcfg.seed},
        {"score_thr", score_thr},
        {"det_vel_sigma", det_vel_sigma},
        {"det_kappa", det_kappa},
        {"det_sigma_curve", det_sigma_curve_arg},
        {"det_gate_curve", det_gate_curve_arg},
        {"dogm", {{"particles_per_box", dcfg.particles_per_box},
                  {"fresh_birth_count", dcfg.fresh_birth_count},
                  {"resample_budget", dcfg.resample_budget},
                  {"sigma_v_proc", dcfg.sigma_v_proc},
                  {"birth_seed_fraction", dcfg.birth_seed_fraction},
                  {"birth_static_fraction", dcfg.birth_static_fraction},
                  {"vel_std_abs", dcfg.vel_std_abs},
                  {"vel_std_rel", dcfg.vel_std_rel},
                  {"resample_neff_ratio", dcfg.resample_neff_ratio},
                  {"birth_mass_share", dcfg.birth_mass_share},
                  {"resolution", dcfg.grid.resolution}}},
        {"tracker", {{"q_pos", tcfg.q_pos},
                     {"vel_max_hits", tcfg.vel_max_hits}, {"vel_conf_min", tcfg.conf_min},
                     {"seed_conf_min", tcfg.seed_conf_min},
                     {"grid_exclude_classes", grid_exclude_arg},
                     {"grid_class_prior", grid_prior_arg},
                     {"q_vel_by_cls", tcfg.q_vel_by_cls}, {"r_pos_by_cls", tcfg.r_pos_by_cls},
                     {"q_vel", tcfg.q_vel},
                     {"r_pos", tcfg.r_pos},
                     {"gate_maha2", tcfg.gate_maha2},
                     {"max_coast_s", tcfg.max_coast_s},
                     {"update_on_sweeps", update_on_sweeps},
                     {"confirm_hits", tcfg.confirm_hits},
                     {"chi2_gate_vel", tcfg.chi2_gate_vel},
                     {"sigma_v_floor", tcfg.sigma_v_floor},
                     {"r_scale", tcfg.r_scale},
                     {"vel_max_age_s", tcfg.vel_max_age_s},
                     {"birth_cov_from_meas", birth_cov_from_meas}}},
        {"runtime_s", secs},
        {"frames", total_frames},
        {"n_dogm_updates", n_dogm_used},
        {"n_detector_updates", n_det_used},
        {"n_tracking_boxes", n_out_boxes},
        {"n_coast_boxes", n_coast_boxes},
        {"coast_output", coast_output},
        {"coast_support_min", coast_support_min},
        {"coast_output_max_s", coast_output_max_s},
        {"det_dir", det_dir},
        {"lidar_cache", lidar_cache}};
    std::ofstream mf(out_dir + "/run_meta.json");
    mf << meta.dump(2) << "\n";
    return 0;
}
