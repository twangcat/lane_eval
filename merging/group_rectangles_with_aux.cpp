#include <vector>
#include <iostream>

#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

using cv::Rect;

class SimilarRects {
public:
    SimilarRects(double _eps) : eps(_eps) {}
    inline bool operator() (const vector<float>& r1, const vector<float>& r2) const {
        double delta = eps * (std::min(r1[2], r2[2]) + std::min(r1[3], r2[3])) * 0.5;
        return std::abs(r1[0] - r2[0]) <= delta &&
            std::abs(r1[1] - r2[1]) <= delta &&
            std::abs(r1[0] + r1[2] - r2[0] - r2[2]) <= delta &&
            std::abs(r1[1] + r1[3] - r2[1] - r2[3]) <= delta;
    }
    double eps;
};

int groupRectanglesWithAux(float* input, int groupThreshold, float eps,
    float* output, int* scores, int nrows, int ncols) {
  if (groupThreshold <= 0 || nrows == 0) {
    return 0;
  }

  vector<vector<float> > rectList(nrows, vector<float>(ncols));
  for (int i = 0; i < nrows; ++i) {
    int base_index = i * ncols;
    for (int j = 0; j < ncols; ++j) {
      if (j < 4) {
        rectList[i][j] = cv::saturate_cast<int>(input[base_index + j]);
      } else {
        rectList[i][j] = input[base_index + j];
      }
    }
  }

  vector<int> labels;
  int nclasses = cv::partition(rectList, labels, SimilarRects(eps));
  vector<vector<float> > frects(nclasses, vector<float>(ncols));
  vector<int> rweights(nclasses, 0);
  int nlabels = (int) labels.size();
  for (int i = 0; i < nlabels; i++)  {
    int cls = labels[i];
    for (int j = 0; j < ncols; ++j) {
      frects[cls][j] += rectList[i][j];
    }
    rweights[cls]++;
  }

  for (int i = 0; i < nclasses; i++) {
    float s = 1.f / rweights[i];
    for (int j = 0; j < ncols; ++j) {
      float scaled = frects[i][j] * s;
      if (j < 4) {
        frects[i][j] = cv::saturate_cast<int>(scaled);
      } else {
        frects[i][j] = scaled;
      }
    }
  }

  int j;
  int num_output = 0;
  for (int i = 0; i < nclasses; i++) {
    int r1 = i;
    int n1 = rweights[i];
    if (n1 <= groupThreshold)
      continue;
    // filter out small face rectangles inside large rectangles
    for (j = 0; j < nclasses; j++) {
      int n2 = rweights[j];

      if (j == i || n2 <= groupThreshold)
        continue;
      int r2 = j;
      int dx = cv::saturate_cast<int>(frects[r2][2] * eps);
      int dy = cv::saturate_cast<int>(frects[r2][3] * eps);

      if (i != j &&
          frects[r1][0] >= frects[r2][0] - dx &&
          frects[r1][1] >= frects[r2][1] - dy &&
          frects[r1][0] + frects[r1][2] <= frects[r2][0] + frects[r2][2] + dx &&
          frects[r1][1] + frects[r1][3] <= frects[r2][1] + frects[r2][3] + dy &&
          (n2 > std::max(3, n1) || n1 < 3))
        break;
    }

    if (j == nclasses) {
      for (int k = 0; k < ncols; ++k) {
        output[num_output * ncols + k] = frects[r1][k];
      }
      scores[num_output] = n1;
      num_output++;
    }
  }

  return num_output;
}
