/**
 * @file
 * @brief The master header file of ThunderKittens. This file includes everything you need!
 */

#pragma once

#if defined(KITTENS_CDNA4)
#include "cdna4/includes.cuh"
#elif defined(KITTENS_CDNA5)
#include "cdna5/includes.cuh"
#elif defined(KITTENS_CDNA3)
#include "cdna3/includes.cuh"
#endif

#include "pyutils/util.cuh"


// #include "pyutils/pyutils.cuh" // for simple binding without including torch