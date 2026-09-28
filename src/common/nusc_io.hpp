// nusc_io.hpp - nuScenes input plumbing shared by the DGMTrack apps.
//
// Everything here is I/O and coordinate bookkeeping: the pose/calibration
// cache, the LiDAR binaries, the detection JSONL, and the global <-> LiDAR
// transforms. No estimation logic lives in this file.
//
// The pose cache (tools/build_pose_cache.py) exists because reading these
// fields from the raw tables costs a full parse of sample_data.json (1.3 GB)
// and ego_pose.json (645 MB) on every run - 3m36s for a single scene, versus
// ~1s from the cache.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "json.hpp"

namespace nusc {

using json = nlohmann::json;

inline Eigen::Quaterniond quat_wxyz(const std::vector<double>& q) {
    return Eigen::Quaterniond(q[0], q[1], q[2], q[3]).normalized();
}
inline double yaw_of(const Eigen::Quaterniond& q) {
    const Eigen::Matrix3d R = q.toRotationMatrix();
    return std::atan2(R(1, 0), R(0, 0));
}
inline double wrap_pi(double a) {
    while (a > M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

// --------------------------------------------------------------- pose cache
struct PoseEntry {
    int64_t ts = 0;
    std::string file, scene;
    Eigen::Vector3d cs_t = Eigen::Vector3d::Zero();
    Eigen::Vector3d ep_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond cs_q{1, 0, 0, 0};
    Eigen::Quaterniond ep_q{1, 0, 0, 0};
};

inline bool load_pose_cache(const std::string& path,
                            std::unordered_map<std::string, PoseEntry>& out) {
    std::ifstream f(path);
    if (!f) return false;
    json j;
    f >> j;
    for (auto it = j["frames"].begin(); it != j["frames"].end(); ++it) {
        const auto& v = it.value();
        PoseEntry e;
        e.ts = v["ts"].get<int64_t>();
        e.file = v["file"].get<std::string>();
        e.scene = v["scene"].get<std::string>();
        const auto ct = v["cs_t"].get<std::vector<double>>();
        const auto et = v["ep_t"].get<std::vector<double>>();
        e.cs_t = Eigen::Vector3d(ct[0], ct[1], ct[2]);
        e.ep_t = Eigen::Vector3d(et[0], et[1], et[2]);
        e.cs_q = quat_wxyz(v["cs_q"].get<std::vector<double>>());
        e.ep_q = quat_wxyz(v["ep_q"].get<std::vector<double>>());
        out.emplace(it.key(), std::move(e));
    }
    return true;
}

// LiDAR (world) pose of a frame: ego pose composed with the sensor extrinsics.
inline void lidar_pose(const PoseEntry& pe, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    const Eigen::Matrix3d R_e = pe.ep_q.toRotationMatrix();
    R = R_e * pe.cs_q.toRotationMatrix();
    t = pe.ep_t + R_e * pe.cs_t;
}

// Rigid transform taking the previous LiDAR frame into the current one.
inline void relative_2d(const Eigen::Matrix3d& R_cur, const Eigen::Vector3d& t_cur,
                        const Eigen::Matrix3d& R_prev, const Eigen::Vector3d& t_prev,
                        Eigen::Matrix2d& R_rel, Eigen::Vector2d& t_rel) {
    const Eigen::Matrix3d R3 = R_cur.transpose() * R_prev;
    const Eigen::Vector3d t3 = R_cur.transpose() * (t_prev - t_cur);
    R_rel << R3(0, 0), R3(0, 1), R3(1, 0), R3(1, 1);
    t_rel << t3.x(), t3.y();
}

// --------------------------------------------------------------- LiDAR bins
inline std::vector<Eigen::Vector2f> load_lidar_xy(const std::string& path) {
    std::vector<Eigen::Vector2f> pts;
    std::ifstream f(path, std::ios::binary);
    if (!f) return pts;
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % 4) return pts;
    std::vector<float> buf(bytes / 4);
    f.read((char*)buf.data(), bytes);
    const size_t nf = buf.size();
    const int stride = (nf % 5 == 0) ? 5 : ((nf % 4 == 0) ? 4 : 6);
    pts.reserve(nf / stride);
    for (size_t i = 0; i + stride <= nf; i += stride)
        pts.emplace_back(buf[i], buf[i + 1]);
    return pts;
}

// Read the pre-extracted 2-D points written by tools/build_lidar_cache.py.
// The tracker only uses x/y inside the grid footprint, so caching just that on
// local storage removes the dataset drive from the inner loop - which
// dominated runtime with the dataset on USB. Returns false if there is no
// cache entry, so the caller can fall back to the raw point cloud.
inline bool load_lidar_xy_cached(const std::string& cache_dir,
                                 const std::string& scene,
                                 const std::string& sd_token,
                                 std::vector<Eigen::Vector2f>& out) {
    if (cache_dir.empty()) return false;
    const std::string path = cache_dir + "/" + scene + "/" + sd_token + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes % (2 * sizeof(float))) return false;
    out.resize(bytes / (2 * sizeof(float)));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return (bool)f;
}

// ---------------------------------------------------------- detection JSONL
struct DetBox {
    std::vector<double> t{0, 0, 0}, s{1, 1, 1}, q{1, 0, 0, 0};
    std::string name, attr;
    double score = 1.0;
    bool has_vel = false;
    double vx = 0, vy = 0;  // detector velocity, GLOBAL frame
};

struct Frame {
    std::string sample_token, sd_token;
    bool is_key = false;
    int64_t ts = -1;
    std::vector<DetBox> boxes;
};

inline const std::vector<std::string>& tracking_classes() {
    static const std::vector<std::string> c = {"car",        "truck",      "bus",
                                               "trailer",    "pedestrian", "motorcycle",
                                               "bicycle"};
    return c;
}
inline int class_id(const std::string& s) {
    const auto& c = tracking_classes();
    for (size_t i = 0; i < c.size(); ++i)
        if (c[i] == s) return (int)i;
    return -1;
}

inline std::vector<Frame> load_scene(const std::string& path) {
    std::vector<Frame> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json j;
        try {
            j = json::parse(line);
        } catch (...) { continue; }
        Frame fr;
        fr.sample_token = j.value("sample_token", std::string());
        fr.sd_token = j.value("sd_token", std::string());
        fr.is_key = j.value("is_key_frame", false);
        fr.ts = j.value("timestamp", (int64_t)-1);
        if (j.contains("boxes"))
            for (const auto& b : j["boxes"]) {
                DetBox d;
                if (b.contains("translation")) d.t = b["translation"].get<std::vector<double>>();
                if (b.contains("size")) d.s = b["size"].get<std::vector<double>>();
                if (b.contains("rotation")) d.q = b["rotation"].get<std::vector<double>>();
                d.name = b.value("detection_name", std::string());
                d.attr = b.value("attribute_name", std::string());
                d.score = b.value("detection_score", 1.0);
                if (b.contains("velocity") && b["velocity"].is_array() &&
                    b["velocity"].size() >= 2) {
                    d.vx = b["velocity"][0].get<double>();
                    d.vy = b["velocity"][1].get<double>();
                    d.has_vel = std::isfinite(d.vx) && std::isfinite(d.vy);
                }
                fr.boxes.push_back(std::move(d));
            }
        out.push_back(std::move(fr));
    }
    std::sort(out.begin(), out.end(),
              [](const Frame& a, const Frame& b) { return a.ts < b.ts; });
    return out;
}

// -------------------------------------------------- global <-> LiDAR frames
struct BoxInLidar {
    double cx = 0, cy = 0, w = 0, l = 0, yaw = 0;
    double vx = 0, vy = 0;   // detector velocity rotated into the LiDAR frame
    bool has_vel = false;
    int cls = -1;
    int orig = -1;           // index into Frame::boxes
};

inline BoxInLidar box_to_lidar(const DetBox& d, const PoseEntry& pe, int orig_idx) {
    const Eigen::Vector3d pg(d.t[0], d.t[1], d.t[2]);
    const Eigen::Quaterniond qg = quat_wxyz(d.q);
    const Eigen::Vector3d p_ego = pe.ep_q.inverse() * (pg - pe.ep_t);
    const Eigen::Vector3d p_l = pe.cs_q.inverse() * (p_ego - pe.cs_t);
    const Eigen::Quaterniond q_l = pe.cs_q.inverse() * (pe.ep_q.inverse() * qg);

    BoxInLidar b;
    b.cx = p_l.x();
    b.cy = p_l.y();
    b.w = d.s[0];
    b.l = d.s[1];
    b.yaw = yaw_of(q_l);
    b.cls = class_id(d.name);
    b.orig = orig_idx;
    if (d.has_vel) {
        const Eigen::Vector3d vg(d.vx, d.vy, 0.0);
        const Eigen::Vector3d v_l = pe.cs_q.inverse() * (pe.ep_q.inverse() * vg);
        b.vx = v_l.x();
        b.vy = v_l.y();
        b.has_vel = true;
    }
    return b;
}

// LiDAR-frame position back to the global frame (inverse of box_to_lidar).
inline Eigen::Vector3d pos_to_global(const PoseEntry& pe, double x, double y, double z) {
    const Eigen::Vector3d p_ego = pe.cs_q * Eigen::Vector3d(x, y, z) + pe.cs_t;
    return pe.ep_q * p_ego + pe.ep_t;
}

inline Eigen::Vector2d vel_to_global(const PoseEntry& pe, const Eigen::Vector2d& v_lidar) {
    const Eigen::Vector3d vl(v_lidar.x(), v_lidar.y(), 0.0);
    const Eigen::Vector3d g = pe.ep_q * (pe.cs_q * vl);
    return Eigen::Vector2d(g.x(), g.y());
}

}  // namespace nusc
