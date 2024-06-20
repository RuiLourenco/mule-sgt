
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
      auto byte_depth = [](int64_t max_value){ return ceil(log2(max_value) / 8); };
      channel = repeat(bind(byte_depth, _r1))[byte_[_val = _val*256 + _1]];

      // convert vector to int tensor: hwc to whc
      auto to_tensor = [](auto&& vec, int w, int h, int c){
        return at::tensor(vec, at::kLong).reshape({h, w, c}).permute({1, 0, 2});
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

  at::Tensor parse_ppm(istream& is) {
    at::Tensor value;

    PPMGrammar<boost::spirit::basic_istream_iterator<char>> ppm_grammar;
    is >> noskipws >> qi::phrase_match(ppm_grammar, qi::ascii::space, value);

    if(is.fail()) throw runtime_error{"PPM parsing error: invalid grammar"};
    if(!is.eof()) throw runtime_error{"PPM Parsing error: file longer than expected"};

    return value;

  }

  at::Tensor read_ppm(istream& is) {

    // copy raw file contents into vector
    const vector<char> bytes{istreambuf_iterator<char>{is}, {}};

    PPMGrammar<vector<char>::const_iterator> ppm_grammar;

    at::Tensor value;
    auto begin = bytes.begin(), end = bytes.end();
    auto good = qi::phrase_parse(begin, end, ppm_grammar, qi::ascii::space, value);

    if(!good) throw runtime_error{"PPM parsing error: invalid grammar"};
    if(begin != end) throw runtime_error{"Parsing error: file longer than expected"};

    return value;
  }

  at::Tensor read_collection(string data_root, string pattern) {

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
        view_list[{u, v}] = entry.path();
      }

    }

    // compute number of views
    int max_u = 0, max_v = 0;
    for (auto&& [u, v] : view_list | adp::map_keys) {
      max_u = max(max_u, u);
      max_v = max(max_v, v);
    }

    // get shape of individual view
    auto first_entry = view_list.cbegin()->second;
    fs::ifstream first_file{first_entry, ios::in | ios::binary};

    auto first_view = read_ppm(first_file);
    const auto view_shape = first_view.sizes();
    const auto view_dtype = first_view.dtype();

    vector<int64_t> lightfield_shape = {max_u + 1, max_v + 1};
    boost::push_back(lightfield_shape, view_shape); // insert view shape as trailing dimension

    auto lightfield = at::empty(lightfield_shape, view_dtype);

    // populate lightfield from read files
    for (auto&& [coord, path] : view_list) {
      auto [u, v] = coord;
      //fs::ifstream file{path, ios::in | ios::binary};
      bio::stream<bio::mapped_file_source> is{path}; // use memory-mapped file for faster transversal
      lightfield.index_put_({u, v}, read_ppm(is));
      std::cout<<u<<" "<<v<<std::endl;
    }


    return lightfield;
  }

}
