// Copyright Ralf W. Grosse-Kunstleve 2002-2004. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace lunus { // LUNUS namespace

  // 
  
  void tohkl(const boost::python::list& M,const boost::python::list& v,boost::python::list& r) 
  {
    boost::python::list l;
    for (int i=1;i<len(v);++i) {
       r[i]=v[i]/2;
    }
  }
}

BOOST_PYTHON_MODULE(lunus)
{
    using namespace boost::python;
    def("tohkl", lunus::tohkl);
}
