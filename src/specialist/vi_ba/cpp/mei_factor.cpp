// MEI (unified-sphere + Brown-Conrady incl. k3) reprojection factors with
// analytical Jacobians. The forward projection mirrors mei_project in
// specialist_camera. Built by specialist.vi_ba._buildext.

#include <gtsam/base/VectorSpace.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>

namespace py = pybind11;
using namespace gtsam;

// Project a lens-frame point through the MEI model, returning the pixel and
// d(pixel)/d(p_lens). False marks the factor inactive: zero norm, behind the
// unit-sphere mirror, or on the far side of the camera from ``obs_bearing``
// (zero disables that check). With xi > 1 the sphere maps points behind the
// camera to finite pixels, giving a frontal corner a behind-camera twin.
static bool mei_project_jac(const Vector3& p_lens, double fx, double fy,
                            double cx, double cy, double k1, double k2, double k3,
                            double p1, double p2, double xi,
                            const Vector3& obs_bearing, Vector2& pixel_pred,
                            Matrix23& j_mei) {
  if (obs_bearing.squaredNorm() > 0.0 && p_lens.dot(obs_bearing) <= 0.0)
    return false;

  const double norm = p_lens.norm();
  if (norm == 0.0) return false;
  const Vector3 s = p_lens / norm;
  const double denom = s(2) + xi;
  if (denom <= 0.0) return false;  // behind the unit-sphere mirror

  const double x = s(0) / denom;
  const double y = s(1) / denom;
  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const double radial_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r2 * r2;
  const double x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

  pixel_pred << fx * x_d + cx, fy * y_d + cy;

  const Vector3 e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1);
  const double inv = 1.0 / (norm * denom);
  const Vector3 dx_dp = (e_x - x * e_z - x * xi * s) * inv;
  const Vector3 dy_dp = (e_y - y * e_z - y * xi * s) * inv;
  const Vector3 dr2_dp = 2.0 * x * dx_dp + 2.0 * y * dy_dp;
  const Vector3 dradial_dp = radial_dr2 * dr2_dp;
  const Vector3 dxy_dp = dx_dp * y + x * dy_dp;
  const Vector3 dxd_dp = dx_dp * radial + x * dradial_dp + 2.0 * p1 * dxy_dp +
                         p2 * (dr2_dp + 4.0 * x * dx_dp);
  const Vector3 dyd_dp = dy_dp * radial + y * dradial_dp +
                         p1 * (dr2_dp + 4.0 * y * dy_dp) + 2.0 * p2 * dxy_dp;
  j_mei.row(0) = fx * dxd_dp.transpose();
  j_mei.row(1) = fy * dyd_dp.transpose();
  return true;
}

static Matrix3 skew(const Vector3& v) {
  Matrix3 m;
  m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return m;
}

// 2x6 Jacobian of a corner's pixel w.r.t. a pose the corner is expressed FROM
// (pose . q), in the GTSAM Pose3 retract convention: right perturbation,
// tangent [omega, v].
static Matrix pose_from_jac(const Matrix23& j_mei_r_lb, const Matrix3& R,
                            const Vector3& q) {
  Matrix H = Matrix::Zero(2, 6);
  H.block(0, 0, 2, 3) = -(j_mei_r_lb * (R * skew(q)));
  H.block(0, 3, 2, 3) = j_mei_r_lb * R;
  return H;
}

// 2x6 Jacobian w.r.t. a pose the corner is transformed TO (pose^-1 . point);
// ``p`` is that point in the pose's own frame.
static Matrix pose_to_jac(const Matrix23& j_mei_r_lb, const Vector3& p) {
  Matrix H = Matrix::Zero(2, 6);
  H.block(0, 0, 2, 3) = j_mei_r_lb * skew(p);
  H.block(0, 3, 2, 3) = -j_mei_r_lb;
  return H;
}

// The camera every factor shares: the body->lens extrinsic, the MEI intrinsics
// and the observed pixel. ``reproject`` maps a body-frame corner to its pixel
// residual and d(pixel)/d(p_body).
struct MeiCorner {
  double fx, fy, cx, cy, k1, k2, k3, p1, p2, xi;
  Vector2 pixel;
  Vector3 obs_bearing;  // tag direction in the lens frame, or zero
  Matrix3 R_lb;         // lens <- body
  Vector3 t_bl;         // body -> lens origin

  MeiCorner(const Pose3& extrinsic, double fx, double fy, double cx, double cy,
            double k1, double k2, double k3, double p1, double p2, double xi,
            const Vector2& pixel, const Vector3& obs_bearing)
      : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), k3(k3), p1(p1), p2(p2),
        xi(xi), pixel(pixel), obs_bearing(obs_bearing),
        R_lb(extrinsic.rotation().matrix().transpose()),
        t_bl(extrinsic.translation()) {}

  bool reproject(const Vector3& p_body, Vector2& residual,
                 Matrix23& j_mei_r_lb) const {
    const Vector3 p_lens = R_lb * (p_body - t_bl);
    Vector2 pixel_pred;
    Matrix23 j_mei;
    if (!mei_project_jac(p_lens, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi,
                         obs_bearing, pixel_pred, j_mei))
      return false;
    residual = pixel_pred - pixel;
    j_mei_r_lb = j_mei * R_lb;
    return true;
  }

  bool reproject_seated(const Rot3& seat, const Vector3& p_body,
                        Vector2& residual, Matrix23& j_body,
                        Matrix23& j_seat) const {
    const Matrix3 R_sb = seat.matrix().transpose();
    const Vector3 p_lens = R_sb * (p_body - t_bl);
    Vector2 pixel_pred;
    Matrix23 j_mei;
    if (!mei_project_jac(p_lens, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi,
                         obs_bearing, pixel_pred, j_mei))
      return false;
    residual = pixel_pred - pixel;
    j_body = j_mei * R_sb;
    j_seat = j_mei * skew(p_lens);
    return true;
  }
};

// The revolute joint the finger factors share: the pivot pose in the base-tag
// frame and the tag corner in the pivot frame at theta = 0. ``corner`` returns
// q = base_T_joint . Rz(theta) . c in the base-tag frame, and dq/dtheta.
struct RevoluteCorner {
  Matrix3 R_bj;  // base_T_joint.rotation()     (base-tag <- joint pivot)
  Vector3 t_bj;  // base_T_joint.translation()  (pivot origin in base-tag)
  Vector3 c;     // tag corner in the pivot frame at theta = 0

  RevoluteCorner(const Pose3& base_T_joint, const Vector3& c)
      : R_bj(base_T_joint.rotation().matrix()),
        t_bj(base_T_joint.translation()),
        c(c) {}

  void corner(double theta, Vector3& q, Vector3& dq_dtheta) const {
    const double ct = std::cos(theta), st = std::sin(theta);
    const Vector3 p_pivot(c(0) * ct - c(1) * st, c(0) * st + c(1) * ct, c(2));
    q = R_bj * p_pivot + t_bj;
    dq_dtheta = R_bj * Vector3(-p_pivot(1), p_pivot(0), 0.0);
  }
};

// Binary on (body_pose, world_landmark): p_body = world_T_body^-1 . landmark.
class MeiFactorCpp : public NoiseModelFactorN<Pose3, Point3> {
  using Base = NoiseModelFactorN<Pose3, Point3>;
  MeiCorner lens_;

 public:
  MeiFactorCpp(Key poseKey, Key pointKey, const SharedNoiseModel& model,
               MeiCorner lens)
      : Base(model, poseKey, pointKey), lens_(std::move(lens)) {}

  Vector evaluateError(const Pose3& pose, const Point3& landmark,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    const Matrix3 R_bw = pose.rotation().matrix().transpose();
    const Vector3 p_b = R_bw * (landmark - pose.translation());

    Vector2 residual;
    Matrix23 j_mei_r_lb;
    if (!lens_.reproject(p_b, residual, j_mei_r_lb)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_mei_r_lb, p_b);
    if (H2) *H2 = j_mei_r_lb * R_bw;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiFactorCpp>(*this);
  }
};

// Binary on (body_pose X, world_T_anchor A) with an optional third key, the
// observing lens's seat Rot3: p_lens = lens_T_body . X^-1 . A . c
class MeiAnchorTagFactorCpp : public NoiseModelFactor {
  MeiCorner lens_;
  Vector3 c_;  // anchor corner in the tag frame

 public:
  MeiAnchorTagFactorCpp(Key poseKey, Key anchorKey, std::optional<Key> seatKey,
                        const SharedNoiseModel& model, MeiCorner lens,
                        const Vector3& c)
      : NoiseModelFactor(model, seatKey ? KeyVector{poseKey, anchorKey, *seatKey}
                                        : KeyVector{poseKey, anchorKey}),
        lens_(std::move(lens)),
        c_(c) {}

  Vector unwhitenedError(const Values& x,
                         OptionalMatrixVecType H = nullptr) const override {
    const Pose3& X = x.at<Pose3>(keys_[0]);
    const Pose3& A = x.at<Pose3>(keys_[1]);
    const Matrix3 R_A = A.rotation().matrix();
    const Vector3 p_world = R_A * c_ + A.translation();
    const Matrix3 R_bw = X.rotation().matrix().transpose();
    const Vector3 p_body = R_bw * (p_world - X.translation());
    const bool seated = keys_.size() == 3;

    Vector2 residual;
    Matrix23 j_body, j_seat;
    const bool active =
        seated ? lens_.reproject_seated(x.at<Rot3>(keys_[2]), p_body, residual,
                                        j_body, j_seat)
               : lens_.reproject(p_body, residual, j_body);
    if (!active) {
      if (H) {
        (*H)[0] = Matrix::Zero(2, 6);
        (*H)[1] = Matrix::Zero(2, 6);
        if (seated) (*H)[2] = Matrix::Zero(2, 3);
      }
      return Vector2::Zero();
    }
    if (H) {
      (*H)[0] = pose_to_jac(j_body, p_body);
      (*H)[1] = pose_from_jac(j_body * R_bw, R_A, c_);
      if (seated) (*H)[2] = j_seat;
    }
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiAnchorTagFactorCpp>(*this);
  }
};

// Seated variants: the lens seat on the body is a Rot3 variable, its translation
// baked from the factory block.

// Binary on (body_T_base_tag, seat rotation).
class MeiSeatedBaseTagFactorCpp : public NoiseModelFactorN<Pose3, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Rot3>;
  MeiCorner lens_;
  Vector3 c_;

 public:
  MeiSeatedBaseTagFactorCpp(Key baseTagKey, Key seatKey,
                            const SharedNoiseModel& model, MeiCorner lens,
                            const Vector3& c)
      : Base(model, baseTagKey, seatKey),
        lens_(std::move(lens)),
        c_(c) {}

  Vector evaluateError(const Pose3& body_T_base_tag, const Rot3& seat,
                       OptionalMatrixType H1,
                       OptionalMatrixType H2) const override {
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = R_bt * c_ + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject_seated(seat, p_body, residual, j_body, j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_from_jac(j_body, R_bt, c_);
    if (H2) *H2 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedBaseTagFactorCpp>(*this);
  }
};

// Ternary on (theta, body_T_base_tag, seat rotation).
class MeiSeatedFingerFactorCpp : public NoiseModelFactorN<double, Pose3, Rot3> {
  using Base = NoiseModelFactorN<double, Pose3, Rot3>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiSeatedFingerFactorCpp(Key thetaKey, Key baseTagKey, Key seatKey,
                           const SharedNoiseModel& model, MeiCorner lens,
                           RevoluteCorner joint)
      : Base(model, thetaKey, baseTagKey, seatKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const double& theta, const Pose3& body_T_base_tag,
                       const Rot3& seat, OptionalMatrixType H1,
                       OptionalMatrixType H2,
                       OptionalMatrixType H3) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 p_body = R_bt * q + body_T_base_tag.translation();

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject_seated(seat, p_body, residual, j_body, j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 1);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = j_body * (R_bt * dq_dtheta);
    if (H2) *H2 = pose_from_jac(j_body, R_bt, q);
    if (H3) *H3 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedFingerFactorCpp>(*this);
  }
};

// Quaternary on (observer X_O, owner X_W, owner body_T_base_tag, observer seat).
class MeiSeatedCrossTagFactorCpp
    : public NoiseModelFactorN<Pose3, Pose3, Pose3, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3, Rot3>;
  MeiCorner lens_;
  Vector3 c_;

 public:
  MeiSeatedCrossTagFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                             Key seatKey, const SharedNoiseModel& model,
                             MeiCorner lens, const Vector3& c)
      : Base(model, observerKey, ownerKey, baseTagKey, seatKey),
        lens_(std::move(lens)),
        c_(c) {}

  Vector evaluateError(const Pose3& X_O, const Pose3& X_W,
                       const Pose3& body_T_base_tag, const Rot3& seat,
                       OptionalMatrixType H1, OptionalMatrixType H2,
                       OptionalMatrixType H3,
                       OptionalMatrixType H4) const override {
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 m = R_bt * c_ + body_T_base_tag.translation();
    const Matrix3 R_W = X_W.rotation().matrix();
    const Vector3 p_world = R_W * m + X_W.translation();
    const Matrix3 R_bwO = X_O.rotation().matrix().transpose();
    const Vector3 p_bodyO = R_bwO * (p_world - X_O.translation());

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject_seated(seat, p_bodyO, residual, j_body, j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      if (H4) *H4 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    if (H1) *H1 = pose_to_jac(j_body, p_bodyO);
    if (H2 || H3) {
      const Matrix23 j_world = j_body * R_bwO;
      if (H2) *H2 = pose_from_jac(j_world, R_W, m);
      if (H3) *H3 = pose_from_jac(j_world * R_W, R_bt, c_);
    }
    if (H4) *H4 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedCrossTagFactorCpp>(*this);
  }
};

// Five keys: the seated cross-tag chain plus the owner's joint angle theta.
class MeiSeatedCrossFingerFactorCpp
    : public NoiseModelFactorN<Pose3, Pose3, Pose3, double, Rot3> {
  using Base = NoiseModelFactorN<Pose3, Pose3, Pose3, double, Rot3>;
  MeiCorner lens_;
  RevoluteCorner joint_;

 public:
  MeiSeatedCrossFingerFactorCpp(Key observerKey, Key ownerKey, Key baseTagKey,
                                Key thetaKey, Key seatKey,
                                const SharedNoiseModel& model, MeiCorner lens,
                                RevoluteCorner joint)
      : Base(model, observerKey, ownerKey, baseTagKey, thetaKey, seatKey),
        lens_(std::move(lens)),
        joint_(std::move(joint)) {}

  Vector evaluateError(const Pose3& X_O, const Pose3& X_W,
                       const Pose3& body_T_base_tag, const double& theta,
                       const Rot3& seat, OptionalMatrixType H1,
                       OptionalMatrixType H2, OptionalMatrixType H3,
                       OptionalMatrixType H4,
                       OptionalMatrixType H5) const override {
    Vector3 q, dq_dtheta;
    joint_.corner(theta, q, dq_dtheta);
    const Matrix3 R_bt = body_T_base_tag.rotation().matrix();
    const Vector3 m = R_bt * q + body_T_base_tag.translation();
    const Matrix3 R_W = X_W.rotation().matrix();
    const Vector3 p_world = R_W * m + X_W.translation();
    const Matrix3 R_bwO = X_O.rotation().matrix().transpose();
    const Vector3 p_bodyO = R_bwO * (p_world - X_O.translation());

    Vector2 residual;
    Matrix23 j_body, j_seat;
    if (!lens_.reproject_seated(seat, p_bodyO, residual, j_body, j_seat)) {
      if (H1) *H1 = Matrix::Zero(2, 6);
      if (H2) *H2 = Matrix::Zero(2, 6);
      if (H3) *H3 = Matrix::Zero(2, 6);
      if (H4) *H4 = Matrix::Zero(2, 1);
      if (H5) *H5 = Matrix::Zero(2, 3);
      return Vector2::Zero();
    }
    const Matrix23 j_world = j_body * R_bwO;
    const Matrix23 j_owner = j_world * R_W;
    if (H1) *H1 = pose_to_jac(j_body, p_bodyO);
    if (H2) *H2 = pose_from_jac(j_world, R_W, m);
    if (H3) *H3 = pose_from_jac(j_owner, R_bt, q);
    if (H4) *H4 = j_owner * (R_bt * dq_dtheta);
    if (H5) *H5 = j_seat;
    return residual;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<MeiSeatedCrossFingerFactorCpp>(*this);
  }
};

// The make_* factories return NoiseModelFactor so the Python object exposes the
// full NoiseModelFactor API; NonlinearFactorGraph.push_back still accepts it.
static NoiseModelFactor::shared_ptr make_mei_factor_cpp(
    size_t poseKey, size_t pointKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& extrinsic, double fx, double fy,
    double cx, double cy, double k1, double k2, double k3, double p1, double p2,
    double xi, const Vector3& obs_bearing) {
  return std::make_shared<MeiFactorCpp>(
      poseKey, pointKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing));
}

static NoiseModelFactor::shared_ptr make_mei_seated_base_tag_factor_cpp(
    size_t baseTagKey, size_t seatKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& extrinsic, const Vector3& c,
    double fx, double fy, double cx, double cy, double k1, double k2, double k3,
    double p1, double p2, double xi, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedBaseTagFactorCpp>(
      baseTagKey, seatKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_seated_finger_factor_cpp(
    size_t thetaKey, size_t baseTagKey, size_t seatKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& extrinsic,
    const Pose3& base_T_joint, const Vector3& c, double fx, double fy,
    double cx, double cy, double k1, double k2, double k3, double p1, double p2,
    double xi, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedFingerFactorCpp>(
      thetaKey, baseTagKey, seatKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

static NoiseModelFactor::shared_ptr make_mei_seated_cross_tag_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, size_t seatKey,
    const Vector2& pixel, const SharedNoiseModel& model, const Pose3& extrinsic,
    const Vector3& c, double fx, double fy, double cx, double cy, double k1,
    double k2, double k3, double p1, double p2, double xi,
    const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedCrossTagFactorCpp>(
      observerKey, ownerKey, baseTagKey, seatKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing),
      c);
}

static NoiseModelFactor::shared_ptr make_mei_seated_cross_finger_factor_cpp(
    size_t observerKey, size_t ownerKey, size_t baseTagKey, size_t thetaKey,
    size_t seatKey, const Vector2& pixel, const SharedNoiseModel& model,
    const Pose3& extrinsic, const Pose3& base_T_joint, const Vector3& c,
    double fx, double fy, double cx, double cy, double k1, double k2, double k3,
    double p1, double p2, double xi, const Vector3& obs_bearing) {
  return std::make_shared<MeiSeatedCrossFingerFactorCpp>(
      observerKey, ownerKey, baseTagKey, thetaKey, seatKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing),
      RevoluteCorner(base_T_joint, c));
}

static NoiseModelFactor::shared_ptr make_mei_anchor_tag_factor_cpp(
    size_t poseKey, size_t anchorKey, const Vector2& pixel,
    const SharedNoiseModel& model, const Pose3& extrinsic, const Vector3& c,
    double fx, double fy, double cx, double cy, double k1, double k2, double k3,
    double p1, double p2, double xi,
    const Vector3& obs_bearing, std::optional<size_t> seatKey) {
  return std::make_shared<MeiAnchorTagFactorCpp>(
      poseKey, anchorKey, seatKey, model,
      MeiCorner(extrinsic, fx, fy, cx, cy, k1, k2, k3, p1, p2, xi, pixel,
                obs_bearing),
      c);
}

PYBIND11_MODULE(_mei_factor_cpp, m) {
  m.doc() = "Native MEI reprojection factors with analytical Jacobians.";
  m.def("make_mei_factor_cpp", &make_mei_factor_cpp, py::arg("pose_key"),
        py::arg("point_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("extrinsic"), py::arg("fx"), py::arg("fy"), py::arg("cx"),
        py::arg("cy"), py::arg("k1"), py::arg("k2"), py::arg("k3"),
        py::arg("p1"), py::arg("p2"), py::arg("xi"), py::arg("obs_bearing"));
  m.def("make_mei_seated_base_tag_factor_cpp",
        &make_mei_seated_base_tag_factor_cpp, py::arg("base_tag_key"),
        py::arg("seat_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("extrinsic"), py::arg("c"), py::arg("fx"), py::arg("fy"),
        py::arg("cx"), py::arg("cy"), py::arg("k1"), py::arg("k2"),
        py::arg("k3"), py::arg("p1"), py::arg("p2"), py::arg("xi"),
        py::arg("obs_bearing"));
  m.def("make_mei_seated_finger_factor_cpp", &make_mei_seated_finger_factor_cpp,
        py::arg("theta_key"), py::arg("base_tag_key"), py::arg("seat_key"),
        py::arg("pixel"), py::arg("noise"), py::arg("extrinsic"),
        py::arg("base_T_joint"), py::arg("c"), py::arg("fx"), py::arg("fy"),
        py::arg("cx"), py::arg("cy"), py::arg("k1"), py::arg("k2"),
        py::arg("k3"), py::arg("p1"), py::arg("p2"), py::arg("xi"),
        py::arg("obs_bearing"));
  m.def("make_mei_seated_cross_tag_factor_cpp",
        &make_mei_seated_cross_tag_factor_cpp, py::arg("observer_key"),
        py::arg("owner_key"), py::arg("base_tag_key"), py::arg("seat_key"),
        py::arg("pixel"), py::arg("noise"), py::arg("extrinsic"), py::arg("c"),
        py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
        py::arg("k1"), py::arg("k2"), py::arg("k3"), py::arg("p1"),
        py::arg("p2"), py::arg("xi"), 
        py::arg("obs_bearing"));
  m.def("make_mei_seated_cross_finger_factor_cpp",
        &make_mei_seated_cross_finger_factor_cpp, py::arg("observer_key"),
        py::arg("owner_key"), py::arg("base_tag_key"), py::arg("theta_key"),
        py::arg("seat_key"), py::arg("pixel"), py::arg("noise"),
        py::arg("extrinsic"), py::arg("base_T_joint"), py::arg("c"),
        py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
        py::arg("k1"), py::arg("k2"), py::arg("k3"), py::arg("p1"),
        py::arg("p2"), py::arg("xi"), py::arg("obs_bearing"));
  m.def("make_mei_anchor_tag_factor_cpp", &make_mei_anchor_tag_factor_cpp,
        py::arg("pose_key"), py::arg("anchor_key"), py::arg("pixel"),
        py::arg("noise"), py::arg("extrinsic"), py::arg("c"), py::arg("fx"),
        py::arg("fy"), py::arg("cx"), py::arg("cy"), py::arg("k1"),
        py::arg("k2"), py::arg("k3"), py::arg("p1"), py::arg("p2"),
        py::arg("xi"),  py::arg("obs_bearing"),
        py::arg("seat_key") = py::none());
}
