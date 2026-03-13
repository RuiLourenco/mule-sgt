/*!
 *    @file  ppm_reader.cpp
 *   @brief implementation of read_ppm
 *
 *  @author  Thadeu Luiz Barbosa Dias and Rui Lourenço
 *
 *  @internal
 *       Created:  11/10/2021
 *      Revision:  none
 *      Compiler:  g++
 *  Organization:  SMT - Signals, Multimedia and Telecommunications Lab
 *     Copyright:  Copyright (c) 2021, Thadeu Luiz Barbosa Dias
 *
 *  This source code is released for free distribution under the terms of the
 *  GNU General Public License version 3 as published by the Free Software Foundation.
 */

#include <string>
#include <stdexcept>
#include <iostream>
#include <map>
#include <limits>

#include <torch/torch.h>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_match.hpp>
#include <boost/phoenix.hpp>

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/iostreams/stream.hpp>

#include <boost/xpressive/xpressive.hpp>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>


using namespace std;
namespace qi = boost::spirit::qi;

namespace fs = boost::filesystem;
namespace bio = boost::iostreams;
namespace xp = boost::xpressive;
namespace adp = boost::adaptors;

namespace {

  template<class Iterator>
  struct PPMGrammar: qi::grammar<Iterator, at::Tensor(), qi::ascii::space_type> {

    PPMGrammar() : PPMGrammar::base_type(start) {
      using namespace qi;
      using namespace qi::labels;

      // concatenate multiple-byte data into single value
      auto byte_depth = [](int64_t max_value){ return 2; };
      channel = repeat(bind(byte_depth, _r1))[byte_[_val = _val*256 + _1]];

      // convert vector to int tensor: hwc to whc
      auto to_tensor = [](auto&& vec, int w, int h, int c){
        return at::tensor(vec, at::kLong).reshape({h, w, c});
      };

      // extract max value, and get channel vector
      // data is disposed in h w c format. make tensor and transpose to h w c
      data_block = int_[_a = _1] >> ascii::space >>
        repeat(_r1 * _r2 * _r3)[channel(_a)][_val = bind(to_tensor, _1, _r1, _r2, _r3)];

      p5 = lit("P5") >> // magic number
        int_[_a = _1] >> int_[_b = _1] >> // width and height
        data_block(_a, _b, 1)[_val = _1]; // single-channel image

      p6 = lit("P6") >> // magic number
        int_[_a = _1] >> int_[_b = _1] >> // width and height
        data_block(_a, _b, 3)[_val = _1]; // rgb

      start %= p5 | p6;
    }

    private:
      qi::rule<Iterator, at::Tensor(), qi::ascii::space_type> start;

      qi::rule<Iterator, int64_t(int)> channel;
      qi::rule<Iterator, at::Tensor(int, int, int), qi::locals<int>> data_block;

      // rules for pgm (p5) and ppm (p6)
      qi::rule<Iterator, at::Tensor(), qi::locals<int, int>, qi::ascii::space_type> p5, p6;
  };

}

namespace io {



/**
 * Reads a PPM image from the given input stream and returns it as a PyTorch tensor.
 *
 * @param is The input stream from which to read the PPM image.
 *
 * @return A PyTorch tensor representing the PPM image. The tensor has dimensions (height, width, channels)
 *         and is transposed to (width, height, channels) before being returned.
 *
 * @throws runtime_error If the PPM image cannot be parsed due to an invalid grammar.
 * @throws runtime_error If the PPM image is longer than expected.
 */
  at::Tensor read_ppm(istream& is) {

    // copy raw file contents into vector
    const vector<char> bytes{istreambuf_iterator<char>{is}, {}};
    //cout<<(int)bytes[0]<<" "<<(int)bytes[1]<<endl;

    PPMGrammar<vector<char>::const_iterator> ppm_grammar;

    at::Tensor value;
    auto begin = bytes.begin(), end = bytes.end();
    auto beginSaved = begin;
    auto good = qi::phrase_parse(begin, end, ppm_grammar, qi::ascii::space, value);
    //cout<<end - beginSaved<<endl;
    //cout<<begin - beginSaved<<endl;
    if(!good) throw runtime_error{"PPM parsing error: invalid grammar"};
    if(begin != end) throw runtime_error{"Parsing error: file longer than expected"};

    return value;
  }

  void write_ppm(const at::Tensor& value, ostream& os) {
    const bool lil_endian = true;
    // convert tensor to vector
    at::Tensor source = value.to(torch::kInt16).cpu().contiguous().view({-1});
    //cout<<"source: "<<source[0].item()<<endl;
    std::vector<int16_t> vec_s(source.data_ptr<int16_t>(), source.data_ptr<int16_t>() + source.numel());
    //cout<<" vec_s = "<<vec_s[0]<<endl;
    std::vector<uint16_t> vec(vec_s.begin(), vec_s.end());
    //cout<<"vec = "<<vec[0]<<endl;
    //Switches endienness for 16bit values!
    if(lil_endian){
      for(auto& n : vec){
        n = (n << 8| n >> 8) ;
      }
    }
    //cout<<"lil'endian vec = "<<vec[0]<<endl;


    const int w = value.size(1);
    const int h = value.size(0);
    const int c = value.size(2);
    const int64_t max_value = pow(2,10) - 1; 
    //cout<<max_value<<endl;
    //Output Magic Number
    os<<"P6"<<" "<<w<<" "<<h<<" "<<max_value<<endl;
    //cout<<vec.size()*sizeof(uint16_t)<<endl;
    os.write((char*)&vec[0],vec.size()*sizeof(uint16_t));

  }

  void write_collection(string data_root, at::Tensor data, std::array<int64_t,2> bias = {0,0}, std::array<int64_t,2> stride = {1,1}) {

    for (int l = 0; l < data.size(0); l++) {
      for (int k = 0; k < data.size(1); k++) {
        std::stringstream filename;
        filename <<data_root<< "/"<<std::setw(3) << std::setfill('0') << k*stride[1]+bias[1]<<"_"<<std::setw(3) << std::setfill('0') << l*stride[0]+bias[0]<<".ppm";
        fs::ofstream os;
        os.open(filename.str(), std::ios::out | std::ios::binary);
        //cout<<"written to ("<<l<<","<<k<<") = "<<data[l][k][0][0][0].item()<<endl;
        write_ppm(data[l][k], os);
        
      }
    }
  }

  /**
   * Reads a collection of data files defined by the given pattern from the specified data root directory.
   *
   * @param data_root The root directory where the data files are located
   * @param pattern The pattern to match the data files
   *
   * @return A torch Tensor representing the collection of data read from the files
   *
   * @throws runtime_error If there are parsing errors or if the file is longer than expected
   */
  at::Tensor read_collection(string data_root, string pattern, int& scale) {

    const auto regex  = xp::sregex::compile(pattern);
    const auto data_dir = fs::path{data_root};

    map<pair<int, int>, fs::path> view_list;

    for (auto&& entry : fs::directory_iterator{data_dir}) {

      const auto name = entry.path().filename();
      xp::smatch what;

      // map file paths to uv coordinate
      if (xp::regex_match(name.string(), what, regex)) {
        int u = stoi(what["U"]);
        int v = stoi(what["V"]);
        view_list[{v, u}] = entry.path();
      }

    }

    if(view_list.empty()) throw runtime_error{"No views found matching pattern"};

    // build sorted unique lists of original coords and a mapping to compact indices
    std::vector<int> uniq_v, uniq_u;
    uniq_v.reserve(view_list.size());
    uniq_u.reserve(view_list.size());
    for (auto&& kv : view_list) {
      uniq_v.push_back(kv.first.first);
      uniq_u.push_back(kv.first.second);
    }
    sort(uniq_v.begin(), uniq_v.end());
    uniq_v.erase(std::unique(uniq_v.begin(), uniq_v.end()), uniq_v.end());
    sort(uniq_u.begin(), uniq_u.end());
    uniq_u.erase(std::unique(uniq_u.begin(), uniq_u.end()), uniq_u.end());

    std::map<int,int> v_map, u_map;
    for (size_t i = 0; i < uniq_v.size(); ++i) v_map[uniq_v[i]] = static_cast<int>(i);
    for (size_t i = 0; i < uniq_u.size(); ++i) u_map[uniq_u[i]] = static_cast<int>(i);

    // get shape of individual view (use first entry)
    auto first_entry = view_list.cbegin()->second;
    fs::ifstream first_file{first_entry, ios::in | ios::binary};
    std::string magic_number;
    int h,w;
    first_file>>magic_number>>w>>h>>scale;
    first_file.clear();
    first_file.seekg(0);
    auto first_view = read_ppm(first_file);
    const auto view_shape = first_view.sizes();
    const auto view_dtype = first_view.dtype();

    // build compact lightfield shape (no gaps)
    vector<int64_t> lightfield_shape = { static_cast<int64_t>(uniq_v.size()), static_cast<int64_t>(uniq_u.size()) };
    boost::push_back(lightfield_shape, view_shape); // insert view shape as trailing dimension

    auto lightfield = at::empty(lightfield_shape, view_dtype);

    // populate lightfield from read files, remapping original coords -> compact indices
    for (auto&& [coord, path] : view_list) {
      auto [v_orig, u_orig] = coord;
      int v = v_map[v_orig];
      int u = u_map[u_orig];
      bio::stream<bio::mapped_file_source> is{path}; // use memory-mapped file for faster transversal
      lightfield.index_put_({v, u}, read_ppm(is));
    }

    return lightfield;
  }

}
