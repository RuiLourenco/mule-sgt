#ifndef IO_H
#define IO_H



namespace at{
    class Tensor;
}

#include <string>

namespace io{

  /**
   * @brief    Reads 4D lightfield from a collection of ppm or pgm files
   *
   * @details  This function reads all files in \p root_path that match a regex pattern defined in \p pattern.
   *           The pattern regex must be compilable with Boost.Xpressive, and must contain two named groups,
   *           \c U and \c V. These named group matches must contain only numbers, and must correspond to the coordinates
   *           of the given match, which must be zero-indexed. Example:
   *           @code
   *             // (group named U matching digits)_(group named V matching digits).ppm
   *             std::string pattern = R"((?P<U>\d+)_(?P<V>\d+)\.ppm)";
   *           @endcode
   *
   *           The dimension of the final axis in the output is \f$1\f$ for pgm and \f$3\f$ for ppm.
   *
   * @param    root_path directory containing all the view ppms for the given lightfield.
   *
   * @param    pattern regex to match and organize the input views.
   *
   * @return   5D tensor of shape \f$[U \times V \times S \times T \times C]\f$.
   */
  at::Tensor read_collection(std::string root_path, std::string pattern, int &scale);

    /**
   * @brief    Read ppm or pgm image from given stream.
   *
   * @details  This function parses pgm or ppm data from the input stream and outputs the corresponding image in the
   *           format \f$[W \times H \times C]\f$, where \f$C=1\f$ for pgm and \f$C=3\f$ for ppm.
   *
   *           This function accepts images with channel depth larger than single byte.
   *           The output tensor dtype is always int64.
   *
   * @param    file input stream.
   *
   * @return   tensor with the parsed image.
   */
  at::Tensor read_ppm(std::istream& file);

  void write_ppm(const at::Tensor& value, std::ostream& os);
  void write_collection(std::string data_root, at::Tensor data);

}

#endif