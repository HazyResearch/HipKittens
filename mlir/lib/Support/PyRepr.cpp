#include "hk/Support/PyRepr.h"

#include "hk/Dialect/HK/IR/HKDialect.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace mlir;
using namespace mlir::hk;

std::string mlir::hk::pyRepr(Type t) {
  auto dn = [](Type e) { return dtypeName(e).str(); };
  if (auto r = dyn_cast<RegTileType>(t))
    return "rt<" + dn(r.getElementType()) + "," + std::to_string(r.getRows()) +
           "x" + std::to_string(r.getCols()) + "," + r.getLayout().str() + ">";
  if (auto s = dyn_cast<SharedTileType>(t)) {
    std::string n = s.getCount() != 1 ? "[" + std::to_string(s.getCount()) + "]"
                                      : "";
    return "st<" + dn(s.getElementType()) + "," + std::to_string(s.getRows()) +
           "x" + std::to_string(s.getCols()) + ">" + n;
  }
  if (auto v = dyn_cast<RegVecType>(t))
    return "rv<" + dn(v.getElementType()) + "," +
           std::to_string(v.getLength()) + "," + v.getKind().str() +
           (v.getUniform() ? ",uniform" : "") + ">";
  if (auto s = dyn_cast<SharedVecType>(t)) {
    std::string n = s.getCount() != 1 ? "[" + std::to_string(s.getCount()) + "]"
                                      : "";
    return "sv<" + dn(s.getElementType()) + "," + std::to_string(s.getLength()) +
           ">" + n;
  }
  if (auto g = dyn_cast<GlobalType>(t))
    return "gl<" + dn(g.getElementType()) + ">";
  if (auto c = dyn_cast<CoordType>(t))
    return std::string("coord<") + (c.getElement() ? "element" : "tile") + ">";
  if (auto g = dyn_cast<GranulesType>(t))
    return "granules<" + pyRepr(g.getTile()) + ">x" +
           std::to_string(g.getNumAddresses());
  if (auto s = dyn_cast<StageType>(t)) {
    std::string d = s.getDepth() != 1 ? "[" + std::to_string(s.getDepth()) + "]"
                                      : "";
    return "stage<" + pyRepr(s.getTile()) + "," + std::to_string(s.getThreads()) +
           "t>" + d + "x" + std::to_string(s.getCalls());
  }
  if (t.isIntOrFloat())
    return "scalar<" + dn(t) + ">";
  return "?";
}

std::string mlir::hk::pyFloat(double v) {
  if (std::isnan(v))
    return "nan";
  if (std::isinf(v))
    return v > 0 ? "inf" : "-inf";
  if (v == 0)
    return std::signbit(v) ? "-0.0" : "0.0";

  // Shortest round-tripping significand, as "d.ddddde+XX".
  char buf[64];
  for (int prec = 1; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*e", prec - 1, v);
    if (std::strtod(buf, nullptr) == v)
      break;
  }
  std::string s(buf);
  bool neg = s[0] == '-';
  if (neg)
    s.erase(0, 1);
  size_t e = s.find('e');
  int exp10 = std::atoi(s.c_str() + e + 1);
  std::string digits;
  for (size_t i = 0; i < e; ++i)
    if (s[i] != '.')
      digits += s[i];
  while (digits.size() > 1 && digits.back() == '0')
    digits.pop_back();

  // value = 0.DIGITS x 10^decpt
  int decpt = exp10 + 1;
  int n = digits.size();
  std::string out;
  if (decpt > -4 && decpt <= 16) {
    if (decpt <= 0)
      out = "0." + std::string(-decpt, '0') + digits;
    else if (decpt >= n)
      out = digits + std::string(decpt - n, '0') + ".0";
    else
      out = digits.substr(0, decpt) + "." + digits.substr(decpt);
  } else {
    out = digits.substr(0, 1);
    if (n > 1)
      out += "." + digits.substr(1);
    int x = decpt - 1;
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", x < 0 ? '-' : '+', std::abs(x));
    out += eb;
  }
  return neg ? "-" + out : out;
}
