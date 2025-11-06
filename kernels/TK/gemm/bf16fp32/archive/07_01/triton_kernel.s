# Triton Kernel Assembly (256x256x64 GEMM, 0 spills)
# Extracted from code.json

   0: ; Begin ATT ASM
   1: ; Addr #0x7fae6df99900
   2: s_add_i32 s1, s9, 0xff
   3: s_ashr_i32 s14, s1, 31
   4: s_lshr_b32 s14, s14, 24
   5: s_add_i32 s1, s1, s14
   6: s_ashr_i32 s1, s1, 8
   7: s_lshl_b32 s14, s1, 2
   8: s_abs_i32 s16, s14
   9: v_cvt_f32_u32_e32 v1, s16
  10: s_ashr_i32 s0, s15, 31
  11: s_lshr_b32 s0, s0, 29
  12: s_add_i32 s0, s15, s0
  13: v_rcp_iflag_f32_e32 v1, v1
  14: s_ashr_i32 s0, s0, 3
  15: s_sub_i32 s17, 0, s16
  16: s_mulk_i32 s15, 0x4c
  17: v_mul_f32_e32 v1, 0x4f7ffffe, v1
  18: v_cvt_u32_f32_e32 v1, v1
  19: s_mulk_i32 s0, 0xfda1
  20: s_add_i32 s0, s0, s15
  21: s_abs_i32 s15, s0
  22: v_readfirstlane_b32 s18, v1
  23: s_mul_i32 s17, s17, s18
  24: s_mul_hi_u32 s17, s18, s17
  25: s_add_i32 s18, s18, s17
  26: s_mul_hi_u32 s17, s15, s18
  27: s_mul_i32 s18, s17, s16
  28: s_xor_b32 s1, s0, s1
  29: s_sub_i32 s15, s15, s18
  30: s_ashr_i32 s1, s1, 31
  31: s_add_i32 s18, s17, 1
  32: s_sub_i32 s19, s15, s16
  33: s_cmp_ge_u32 s15, s16
  34: s_cselect_b32 s17, s18, s17
  35: s_cselect_b32 s15, s19, s15
  36: s_add_i32 s18, s17, 1
  37: s_cmp_ge_u32 s15, s16
  38: s_cselect_b32 s15, s18, s17
  39: s_add_i32 s16, s8, 0xff
  40: s_ashr_i32 s17, s16, 31
  41: s_xor_b32 s15, s15, s1
  42: s_lshr_b32 s17, s17, 24
  43: s_sub_i32 s1, s15, s1
  44: s_add_i32 s16, s16, s17
  45: s_lshl_b32 s15, s1, 2
  46: s_ashr_i32 s16, s16, 8
  47: s_sub_i32 s16, s16, s15
  48: s_min_i32 s16, s16, 4
  49: s_abs_i32 s17, s16
  50: v_cvt_f32_u32_e32 v1, s17
  51: s_sub_i32 s18, 0, s17
  52: s_mul_i32 s1, s1, s14
  53: s_sub_i32 s0, s0, s1
  54: v_rcp_iflag_f32_e32 v1, v1
  55: s_abs_i32 s14, s0
  56: s_xor_b32 s1, s0, s16
  57: s_ashr_i32 s1, s1, 31
  58: v_mul_f32_e32 v1, 0x4f7ffffe, v1
  59: v_cvt_u32_f32_e32 v1, v1
  60: v_lshlrev_b32_e32 v2, 3, v0
  61: v_and_b32_e32 v34, 56, v2
  62: v_bfrev_b32_e32 v30, 1
  63: v_readfirstlane_b32 s19, v1
  64: s_mul_i32 s18, s18, s19
  65: s_mul_hi_u32 s18, s19, s18
  66: s_add_i32 s19, s19, s18
  67: s_mul_hi_u32 s18, s14, s19
  68: s_mul_i32 s19, s18, s17
  69: s_sub_i32 s14, s14, s19
  70: s_add_i32 s19, s18, 1
  71: s_sub_i32 s20, s14, s17
  72: s_cmp_ge_u32 s14, s17
  73: s_cselect_b32 s18, s19, s18
  74: s_cselect_b32 s14, s20, s14
  75: s_add_i32 s19, s18, 1
  76: s_cmp_ge_u32 s14, s17
  77: s_cselect_b32 s14, s19, s18
  78: s_xor_b32 s14, s14, s1
  79: s_sub_i32 s20, s14, s1
  80: s_mul_i32 s1, s20, s16
  81: s_sub_i32 s23, s0, s1
  82: s_add_i32 s23, s23, s15
  83: s_lshl_b32 s22, s23, 8
  84: s_mul_i32 s0, s22, s11
  85: s_ashr_i32 s1, s0, 31
  86: s_lshl_b32 s21, s11, 6
  87: s_lshl_b64 s[0:1], s[0:1], 1
  88: s_add_u32 s16, s2, s0
  89: v_lshrrev_b32_e32 v1, 3, v0
  90: s_addc_u32 s17, s3, s1
  91: s_add_i32 s24, s10, 63
  92: v_mad_u64_u32 v[2:3], s[0:1], s11, v1, v[34:35]
  93: s_cmp_gt_i32 s24, 63
  94: v_add_u32_e32 v3, s21, v2
  95: s_cselect_b64 vcc, -1, 0
  96: s_cmp_lt_i32 s24, 64
  97: v_add_u32_e32 v10, s21, v3
  98: s_cselect_b64 s[14:15], -1, 0
  99: s_lshl_b32 s10, s20, 8
 100: v_lshlrev_b32_e32 v146, 1, v2
 101: v_lshlrev_b32_e32 v147, 1, v3
 102: v_lshlrev_b32_e32 v148, 1, v10
 103: v_add_lshl_u32 v149, v10, s21, 1
 104: s_mul_i32 s0, s10, s12
 105: s_and_b32 s17, s17, 0xffff
 106: s_mov_b32 s19, 0x27000
 107: s_mov_b32 s18, 0x7ffffffe
 108: v_cndmask_b32_e32 v11, v30, v146, vcc
 109: v_cndmask_b32_e32 v12, v30, v147, vcc
 110: v_cndmask_b32_e32 v18, v30, v148, vcc
 111: v_cndmask_b32_e32 v19, v30, v149, vcc
 112: s_ashr_i32 s1, s0, 31
 113: buffer_load_dwordx4 v[2:5], v11, s[16:19], 0 offen
 114: buffer_load_dwordx4 v[6:9], v12, s[16:19], 0 offen
 115: s_nop 0
 116: buffer_load_dwordx4 v[10:13], v18, s[16:19], 0 offen
 117: buffer_load_dwordx4 v[14:17], v19, s[16:19], 0 offen
 118: s_lshl_b32 s25, s12, 6
 119: s_lshl_b64 s[20:21], s[0:1], 1
 120: v_mad_u64_u32 v[18:19], s[0:1], s12, v1, v[34:35]
 121: v_add_u32_e32 v19, s25, v18
 122: s_add_u32 s16, s4, s20
 123: v_add_u32_e32 v26, s25, v19
 124: s_addc_u32 s17, s5, s21
 125: v_lshlrev_b32_e32 v152, 1, v18
 126: v_lshlrev_b32_e32 v153, 1, v19
 127: v_lshlrev_b32_e32 v154, 1, v26
 128: s_and_b32 s17, s17, 0xffff
 129: v_cndmask_b32_e32 v18, v30, v152, vcc
 130: v_cndmask_b32_e32 v22, v30, v153, vcc
 131: v_cndmask_b32_e32 v27, v30, v154, vcc
 132: buffer_load_dwordx4 v[18:21], v18, s[16:19], 0 offen
 133: v_add_lshl_u32 v155, v26, s25, 1
 134: buffer_load_dwordx4 v[22:25], v22, s[16:19], 0 offen
 135: v_cndmask_b32_e32 v30, v30, v155, vcc
 136: buffer_load_dwordx4 v[26:29], v27, s[16:19], 0 offen
 137: v_lshrrev_b32_e32 v35, 1, v0
 138: buffer_load_dwordx4 v[30:33], v30, s[16:19], 0 offen
 139: v_and_b32_e32 v36, 56, v35
 140: v_and_b32_e32 v37, 60, v35
 141: v_and_or_b32 v35, v35, 4, v34
 142: v_or_b32_e32 v34, 4, v34
 143: v_lshlrev_b32_e32 v1, 6, v1
 144: v_xor_b32_e32 v35, v36, v35
 145: v_xor_b32_e32 v34, v37, v34
 146: v_or_b32_e32 v35, v35, v1
 147: v_or_b32_e32 v1, v34, v1
 148: v_lshlrev_b32_e32 v1, 1, v1
 149: s_add_i32 s12, 0, 0x8000
 150: v_add_u32_e32 v157, 0, v1
 151: v_add_u32_e32 v159, s12, v1
 152: v_add_u32_e32 v1, 0xff, v0
 153: s_movk_i32 s0, 0x1ff
 154: v_lshlrev_b32_e32 v34, 1, v35
 155: v_cmp_gt_u32_e32 vcc, s0, v1
 156: s_movk_i32 s0, 0x1fe
 157: v_add_u32_e32 v156, 0, v34
 158: v_add_u32_e32 v158, s12, v34
 159: v_cmp_lt_u32_e64 s[0:1], s0, v1
 160: s_waitcnt vmcnt(7)
 161: ds_write_b64 v156, v[2:3]
 162: s_waitcnt vmcnt(6)
 163: ds_write_b64 v156, v[6:7] offset:8192
 164: ds_write2st64_b64 v157, v[4:5], v[8:9] offset1:16
 165: s_waitcnt vmcnt(5)
 166: ds_write_b64 v156, v[10:11] offset:16384
 167: s_waitcnt vmcnt(4)
 168: ds_write_b64 v156, v[14:15] offset:24576
 169: ds_write2st64_b64 v157, v[12:13], v[16:17] offset0:32 offset1:48
 170: s_waitcnt vmcnt(3)
 171: ds_write_b64 v156, v[18:19] offset:32768
 172: ds_write_b64 v157, v[20:21] offset:32768
 173: s_waitcnt vmcnt(2)
 174: ds_write_b64 v158, v[22:23] offset:8192
 175: s_waitcnt vmcnt(1)
 176: ds_write_b64 v158, v[26:27] offset:16384
 177: ds_write2st64_b64 v159, v[24:25], v[28:29] offset0:16 offset1:32
 178: s_waitcnt vmcnt(0)
 179: ds_write_b64 v158, v[30:31] offset:24576
 180: ds_write_b64 v159, v[32:33] offset:24576
 181: s_waitcnt lgkmcnt(0)
 182: s_barrier
 183: s_and_saveexec_b64 s[16:17], s[0:1]
 184: s_cbranch_execz 1
 185: s_or_b64 exec, exec, s[16:17]
 186: v_lshrrev_b32_e32 v1, 6, v0
 187: v_mov_b32_e32 v5, 0
 188: s_cmpk_lt_i32 s24, 0x80
 189: v_lshlrev_b32_e32 v1, 4, v1
 190: v_mov_b32_e32 v4, v5
 191: v_mov_b32_e32 v3, v5
 192: v_mov_b32_e32 v2, v5
 193: v_mov_b32_e32 v9, v5
 194: v_mov_b32_e32 v8, v5
 195: v_mov_b32_e32 v7, v5
 196: v_mov_b32_e32 v6, v5
 197: v_mov_b32_e32 v13, v5
 198: v_mov_b32_e32 v12, v5
 199: v_mov_b32_e32 v11, v5
 200: v_mov_b32_e32 v10, v5
 201: v_mov_b32_e32 v17, v5
 202: v_mov_b32_e32 v16, v5
 203: v_mov_b32_e32 v15, v5
 204: v_mov_b32_e32 v14, v5
 205: v_mov_b32_e32 v21, v5
 206: v_mov_b32_e32 v20, v5
 207: v_mov_b32_e32 v19, v5
 208: v_mov_b32_e32 v18, v5
 209: v_mov_b32_e32 v25, v5
 210: v_mov_b32_e32 v24, v5
 211: v_mov_b32_e32 v23, v5
 212: v_mov_b32_e32 v22, v5
 213: v_mov_b32_e32 v29, v5
 214: v_mov_b32_e32 v28, v5
 215: v_mov_b32_e32 v27, v5
 216: v_mov_b32_e32 v26, v5
 217: v_mov_b32_e32 v33, v5
 218: v_mov_b32_e32 v32, v5
 219: v_mov_b32_e32 v31, v5
 220: v_mov_b32_e32 v30, v5
 221: v_mov_b32_e32 v37, v5
 222: v_mov_b32_e32 v36, v5
 223: v_mov_b32_e32 v35, v5
 224: v_mov_b32_e32 v34, v5
 225: v_mov_b32_e32 v41, v5
 226: v_mov_b32_e32 v40, v5
 227: v_mov_b32_e32 v39, v5
 228: v_mov_b32_e32 v38, v5
 229: v_mov_b32_e32 v45, v5
 230: v_mov_b32_e32 v44, v5
 231: v_mov_b32_e32 v43, v5
 232: v_mov_b32_e32 v42, v5
 233: v_mov_b32_e32 v49, v5
 234: v_mov_b32_e32 v48, v5
 235: v_mov_b32_e32 v47, v5
 236: v_mov_b32_e32 v46, v5
 237: v_mov_b32_e32 v53, v5
 238: v_mov_b32_e32 v52, v5
 239: v_mov_b32_e32 v51, v5
 240: v_mov_b32_e32 v50, v5
 241: v_mov_b32_e32 v57, v5
 242: v_mov_b32_e32 v56, v5
 243: v_mov_b32_e32 v55, v5
 244: v_mov_b32_e32 v54, v5
 245: v_mov_b32_e32 v61, v5
 246: v_mov_b32_e32 v60, v5
 247: v_mov_b32_e32 v59, v5
 248: v_mov_b32_e32 v58, v5
 249: v_mov_b32_e32 v65, v5
 250: v_mov_b32_e32 v64, v5
 251: v_mov_b32_e32 v63, v5
 252: v_mov_b32_e32 v62, v5
 253: v_mov_b32_e32 v69, v5
 254: v_mov_b32_e32 v68, v5
 255: v_mov_b32_e32 v67, v5
 256: v_mov_b32_e32 v66, v5
 257: v_mov_b32_e32 v73, v5
 258: v_mov_b32_e32 v72, v5
 259: v_mov_b32_e32 v71, v5
 260: v_mov_b32_e32 v70, v5
 261: v_mov_b32_e32 v77, v5
 262: v_mov_b32_e32 v76, v5
 263: v_mov_b32_e32 v75, v5
 264: v_mov_b32_e32 v74, v5
 265: v_mov_b32_e32 v81, v5
 266: v_mov_b32_e32 v80, v5
 267: v_mov_b32_e32 v79, v5
 268: v_mov_b32_e32 v78, v5
 269: v_mov_b32_e32 v85, v5
 270: v_mov_b32_e32 v84, v5
 271: v_mov_b32_e32 v83, v5
 272: v_mov_b32_e32 v82, v5
 273: v_mov_b32_e32 v89, v5
 274: v_mov_b32_e32 v88, v5
 275: v_mov_b32_e32 v87, v5
 276: v_mov_b32_e32 v86, v5
 277: v_mov_b32_e32 v93, v5
 278: v_mov_b32_e32 v92, v5
 279: v_mov_b32_e32 v91, v5
 280: v_mov_b32_e32 v90, v5
 281: v_mov_b32_e32 v97, v5
 282: v_mov_b32_e32 v96, v5
 283: v_mov_b32_e32 v95, v5
 284: v_mov_b32_e32 v94, v5
 285: v_mov_b32_e32 v101, v5
 286: v_mov_b32_e32 v100, v5
 287: v_mov_b32_e32 v99, v5
 288: v_mov_b32_e32 v98, v5
 289: v_mov_b32_e32 v105, v5
 290: v_mov_b32_e32 v104, v5
 291: v_mov_b32_e32 v103, v5
 292: v_mov_b32_e32 v102, v5
 293: v_mov_b32_e32 v109, v5
 294: v_mov_b32_e32 v108, v5
 295: v_mov_b32_e32 v107, v5
 296: v_mov_b32_e32 v106, v5
 297: v_mov_b32_e32 v113, v5
 298: v_mov_b32_e32 v112, v5
 299: v_mov_b32_e32 v111, v5
 300: v_mov_b32_e32 v110, v5
 301: v_mov_b32_e32 v117, v5
 302: v_mov_b32_e32 v116, v5
 303: v_mov_b32_e32 v115, v5
 304: v_mov_b32_e32 v114, v5
 305: v_mov_b32_e32 v121, v5
 306: v_mov_b32_e32 v120, v5
 307: v_mov_b32_e32 v119, v5
 308: v_mov_b32_e32 v118, v5
 309: v_mov_b32_e32 v125, v5
 310: v_mov_b32_e32 v124, v5
 311: v_mov_b32_e32 v123, v5
 312: v_mov_b32_e32 v122, v5
 313: v_mov_b32_e32 v129, v5
 314: v_mov_b32_e32 v128, v5
 315: v_mov_b32_e32 v127, v5
 316: v_mov_b32_e32 v126, v5
 317: v_lshrrev_b32_e32 v150, 4, v0
 318: v_and_b32_e32 v151, 15, v0
 319: v_lshrrev_b32_e32 v160, 2, v0
 320: s_cbranch_scc1 609
 321: v_and_b32_e32 v2, 12, v160
 322: v_and_or_b32 v5, v1, 48, v151
 323: v_lshlrev_b32_e32 v6, 2, v151
 324: v_xor_b32_e32 v7, v2, v6
 325: v_lshlrev_b32_e32 v5, 6, v5
 326: v_or_b32_e32 v7, v5, v7
 327: v_bfe_u32 v4, v0, 4, 2
 328: v_lshlrev_b32_e32 v7, 1, v7
 329: s_ashr_i32 s0, s24, 31
 330: v_and_or_b32 v3, v150, 16, v151
 331: v_add_u32_e32 v161, 0, v7
 332: v_add_u32_e32 v162, s12, v7
 333: v_or_b32_e32 v7, 4, v4
 334: s_lshr_b32 s0, s0, 26
 335: v_lshlrev_b32_e32 v3, 7, v3
 336: v_xor_b32_e32 v7, v7, v151
 337: s_add_i32 s24, s24, s0
 338: v_lshl_or_b32 v7, v7, 3, v3
 339: s_ashr_i32 s0, s24, 6
 340: v_add_u32_e32 v163, 0, v7
 341: v_or_b32_e32 v7, 16, v2
 342: v_xor_b32_e32 v7, v7, v6
 343: s_max_i32 s17, s0, 2
 344: v_or_b32_e32 v7, v5, v7
 345: s_add_u32 s0, s4, s20
 346: v_lshlrev_b32_e32 v7, 1, v7
 347: s_addc_u32 s1, s5, s21
 348: v_add_u32_e32 v164, 0, v7
 349: v_add_u32_e32 v165, s12, v7
 350: v_or_b32_e32 v7, 8, v4
 351: s_add_u32 s4, s0, 0x80
 352: s_mul_i32 s11, s11, s23
 353: v_xor_b32_e32 v7, v7, v151
 354: s_addc_u32 s5, s1, 0
 355: s_lshl_b32 s0, s11, 8
 356: v_lshl_or_b32 v7, v7, 3, v3
 357: s_ashr_i32 s1, s0, 31
 358: v_add_u32_e32 v166, 0, v7
 359: v_or_b32_e32 v7, 32, v2
 360: v_or_b32_e32 v2, 48, v2
 361: s_lshl_b64 s[0:1], s[0:1], 1
 362: v_xor_b32_e32 v0, v4, v151
 363: v_xor_b32_e32 v7, v7, v6
 364: v_or_b32_e32 v4, 12, v4
 365: v_xor_b32_e32 v2, v2, v6
 366: s_add_u32 s0, s0, s2
 367: v_or_b32_e32 v7, v5, v7
 368: v_xor_b32_e32 v4, v4, v151
 369: v_or_b32_e32 v2, v5, v2
 370: s_addc_u32 s1, s1, s3
 371: v_lshl_or_b32 v0, v0, 3, v3
 372: v_lshlrev_b32_e32 v7, 1, v7
 373: v_lshl_or_b32 v3, v4, 3, v3
 374: v_lshlrev_b32_e32 v2, 1, v2
 375: s_add_u32 s11, s0, 0x80
 376: v_mov_b32_e32 v126, 0
 377: v_add_u32_e32 v0, 0, v0
 378: v_add_u32_e32 v167, 0, v7
 379: v_add_u32_e32 v168, s12, v7
 380: v_add_u32_e32 v169, 0, v3
 381: v_add_u32_e32 v170, 0, v2
 382: v_add_u32_e32 v171, s12, v2
 383: s_addc_u32 s16, s1, 0
 384: s_add_i32 s17, s17, -1
 385: s_mov_b32 s3, 0x27000
 386: s_mov_b32 s2, 0x7ffffffe
 387: v_mov_b32_e32 v127, v126
 388: v_mov_b32_e32 v128, v126
 389: v_mov_b32_e32 v129, v126
 390: v_mov_b32_e32 v122, v126
 391: v_mov_b32_e32 v123, v126
 392: v_mov_b32_e32 v124, v126
 393: v_mov_b32_e32 v125, v126
 394: v_mov_b32_e32 v118, v126
 395: v_mov_b32_e32 v119, v126
 396: v_mov_b32_e32 v120, v126
 397: v_mov_b32_e32 v121, v126
 398: v_mov_b32_e32 v114, v126
 399: v_mov_b32_e32 v115, v126
 400: v_mov_b32_e32 v116, v126
 401: v_mov_b32_e32 v117, v126
 402: v_mov_b32_e32 v110, v126
 403: v_mov_b32_e32 v111, v126
 404: v_mov_b32_e32 v112, v126
 405: v_mov_b32_e32 v113, v126
 406: v_mov_b32_e32 v106, v126
 407: v_mov_b32_e32 v107, v126
 408: v_mov_b32_e32 v108, v126
 409: v_mov_b32_e32 v109, v126
 410: v_mov_b32_e32 v102, v126
 411: v_mov_b32_e32 v103, v126
 412: v_mov_b32_e32 v104, v126
 413: v_mov_b32_e32 v105, v126
 414: v_mov_b32_e32 v98, v126
 415: v_mov_b32_e32 v99, v126
 416: v_mov_b32_e32 v100, v126
 417: v_mov_b32_e32 v101, v126
 418: v_mov_b32_e32 v94, v126
 419: v_mov_b32_e32 v95, v126
 420: v_mov_b32_e32 v96, v126
 421: v_mov_b32_e32 v97, v126
 422: v_mov_b32_e32 v90, v126
 423: v_mov_b32_e32 v91, v126
 424: v_mov_b32_e32 v92, v126
 425: v_mov_b32_e32 v93, v126
 426: v_mov_b32_e32 v86, v126
 427: v_mov_b32_e32 v87, v126
 428: v_mov_b32_e32 v88, v126
 429: v_mov_b32_e32 v89, v126
 430: v_mov_b32_e32 v82, v126
 431: v_mov_b32_e32 v83, v126
 432: v_mov_b32_e32 v84, v126
 433: v_mov_b32_e32 v85, v126
 434: v_mov_b32_e32 v78, v126
 435: v_mov_b32_e32 v79, v126
 436: v_mov_b32_e32 v80, v126
 437: v_mov_b32_e32 v81, v126
 438: v_mov_b32_e32 v74, v126
 439: v_mov_b32_e32 v75, v126
 440: v_mov_b32_e32 v76, v126
 441: v_mov_b32_e32 v77, v126
 442: v_mov_b32_e32 v70, v126
 443: v_mov_b32_e32 v71, v126
 444: v_mov_b32_e32 v72, v126
 445: v_mov_b32_e32 v73, v126
 446: v_mov_b32_e32 v66, v126
 447: v_mov_b32_e32 v67, v126
 448: v_mov_b32_e32 v68, v126
 449: v_mov_b32_e32 v69, v126
 450: v_mov_b32_e32 v62, v126
 451: v_mov_b32_e32 v63, v126
 452: v_mov_b32_e32 v64, v126
 453: v_mov_b32_e32 v65, v126
 454: v_mov_b32_e32 v58, v126
 455: v_mov_b32_e32 v59, v126
 456: v_mov_b32_e32 v60, v126
 457: v_mov_b32_e32 v61, v126
 458: v_mov_b32_e32 v54, v126
 459: v_mov_b32_e32 v55, v126
 460: v_mov_b32_e32 v56, v126
 461: v_mov_b32_e32 v57, v126
 462: v_mov_b32_e32 v50, v126
 463: v_mov_b32_e32 v51, v126
 464: v_mov_b32_e32 v52, v126
 465: v_mov_b32_e32 v53, v126
 466: v_mov_b32_e32 v46, v126
 467: v_mov_b32_e32 v47, v126
 468: v_mov_b32_e32 v48, v126
 469: v_mov_b32_e32 v49, v126
 470: v_mov_b32_e32 v42, v126
 471: v_mov_b32_e32 v43, v126
 472: v_mov_b32_e32 v44, v126
 473: v_mov_b32_e32 v45, v126
 474: v_mov_b32_e32 v38, v126
 475: v_mov_b32_e32 v39, v126
 476: v_mov_b32_e32 v40, v126
 477: v_mov_b32_e32 v41, v126
 478: v_mov_b32_e32 v34, v126
 479: v_mov_b32_e32 v35, v126
 480: v_mov_b32_e32 v36, v126
 481: v_mov_b32_e32 v37, v126
 482: v_mov_b32_e32 v30, v126
 483: v_mov_b32_e32 v31, v126
 484: v_mov_b32_e32 v32, v126
 485: v_mov_b32_e32 v33, v126
 486: v_mov_b32_e32 v26, v126
 487: v_mov_b32_e32 v27, v126
 488: v_mov_b32_e32 v28, v126
 489: v_mov_b32_e32 v29, v126
 490: v_mov_b32_e32 v22, v126
 491: v_mov_b32_e32 v23, v126
 492: v_mov_b32_e32 v24, v126
 493: v_mov_b32_e32 v25, v126
 494: v_mov_b32_e32 v18, v126
 495: v_mov_b32_e32 v19, v126
 496: v_mov_b32_e32 v20, v126
 497: v_mov_b32_e32 v21, v126
 498: v_mov_b32_e32 v14, v126
 499: v_mov_b32_e32 v15, v126
 500: v_mov_b32_e32 v16, v126
 501: v_mov_b32_e32 v17, v126
 502: v_mov_b32_e32 v10, v126
 503: v_mov_b32_e32 v11, v126
 504: v_mov_b32_e32 v12, v126
 505: v_mov_b32_e32 v13, v126
 506: v_mov_b32_e32 v6, v126
 507: v_mov_b32_e32 v7, v126
 508: v_mov_b32_e32 v8, v126
 509: v_mov_b32_e32 v9, v126
 510: v_mov_b32_e32 v2, v126
 511: v_mov_b32_e32 v3, v126
 512: v_mov_b32_e32 v4, v126
 513: v_mov_b32_e32 v5, v126
 514: s_and_b32 s1, s16, 0xffff
 515: s_mov_b32 s0, s11
 516: buffer_load_dwordx4 v[134:137], v146, s[0:3], 0 offen
 517: buffer_load_dwordx4 v[142:145], v147, s[0:3], 0 offen
 518: buffer_load_dwordx4 v[130:133], v148, s[0:3], 0 offen
 519: buffer_load_dwordx4 v[138:141], v149, s[0:3], 0 offen
 520: ds_read2st64_b64 v[172:175], v0 offset1:8
 521: ds_read2st64_b64 v[176:179], v0 offset0:16 offset1:24
 522: ds_read2st64_b64 v[180:183], v0 offset0:32 offset1:40
 523: ds_read2st64_b64 v[184:187], v0 offset0:48 offset1:56
 524: ds_read2st64_b64 v[188:191], v162 offset0:16 offset1:32
 525: ds_read_b64 v[192:193], v161 offset:32768
 526: ds_read_b64 v[194:195], v162 offset:24576
 527: s_waitcnt lgkmcnt(0)
 528: s_barrier
 529: s_waitcnt lgkmcnt(0)
 530: v_mfma_f32_16x16x16_f16 v[126:129], v[192:193], v[172:173], v[126:129]
 531: s_setprio 1
 532: v_mfma_f32_16x16x16_f16 v[122:125], v[188:189], v[172:173], v[122:125]
 533: v_mfma_f32_16x16x16_f16 v[118:121], v[190:191], v[172:173], v[118:121]
 534: v_mfma_f32_16x16x16_f16 v[114:117], v[194:195], v[172:173], v[114:117]
 535: v_mfma_f32_16x16x16_f16 v[110:113], v[192:193], v[174:175], v[110:113]
 536: v_mfma_f32_16x16x16_f16 v[106:109], v[188:189], v[174:175], v[106:109]
 537: v_mfma_f32_16x16x16_f16 v[102:105], v[190:191], v[174:175], v[102:105]
 538: v_mfma_f32_16x16x16_f16 v[98:101], v[194:195], v[174:175], v[98:101]
 539: v_mfma_f32_16x16x16_f16 v[94:97], v[192:193], v[176:177], v[94:97]
 540: v_mfma_f32_16x16x16_f16 v[90:93], v[188:189], v[176:177], v[90:93]
 541: v_mfma_f32_16x16x16_f16 v[86:89], v[190:191], v[176:177], v[86:89]
 542: v_mfma_f32_16x16x16_f16 v[82:85], v[194:195], v[176:177], v[82:85]
 543: v_mfma_f32_16x16x16_f16 v[78:81], v[192:193], v[178:179], v[78:81]
 544: v_mfma_f32_16x16x16_f16 v[74:77], v[188:189], v[178:179], v[74:77]
 545: v_mfma_f32_16x16x16_f16 v[70:73], v[190:191], v[178:179], v[70:73]
 546: v_mfma_f32_16x16x16_f16 v[66:69], v[194:195], v[178:179], v[66:69]
 547: v_mfma_f32_16x16x16_f16 v[62:65], v[192:193], v[180:181], v[62:65]
 548: v_mfma_f32_16x16x16_f16 v[58:61], v[188:189], v[180:181], v[58:61]
 549: v_mfma_f32_16x16x16_f16 v[54:57], v[190:191], v[180:181], v[54:57]
 550: v_mfma_f32_16x16x16_f16 v[50:53], v[194:195], v[180:181], v[50:53]
 551: v_mfma_f32_16x16x16_f16 v[46:49], v[192:193], v[182:183], v[46:49]
 552: v_mfma_f32_16x16x16_f16 v[42:45], v[188:189], v[182:183], v[42:45]
 553: v_mfma_f32_16x16x16_f16 v[38:41], v[190:191], v[182:183], v[38:41]
 554: v_mfma_f32_16x16x16_f16 v[34:37], v[194:195], v[182:183], v[34:37]
 555: v_mfma_f32_16x16x16_f16 v[30:33], v[192:193], v[184:185], v[30:33]
 556: v_mfma_f32_16x16x16_f16 v[26:29], v[188:189], v[184:185], v[26:29]
 557: v_mfma_f32_16x16x16_f16 v[22:25], v[190:191], v[184:185], v[22:25]
 558: v_mfma_f32_16x16x16_f16 v[18:21], v[194:195], v[184:185], v[18:21]
 559: v_mfma_f32_16x16x16_f16 v[14:17], v[192:193], v[186:187], v[14:17]
 560: v_mfma_f32_16x16x16_f16 v[10:13], v[188:189], v[186:187], v[10:13]
 561: v_mfma_f32_16x16x16_f16 v[6:9], v[190:191], v[186:187], v[6:9]
 562: v_mfma_f32_16x16x16_f16 v[2:5], v[194:195], v[186:187], v[2:5]
 563: s_setprio 0
 564: s_barrier
 565: s_waitcnt lgkmcnt(0)
 566: s_and_b32 s1, s5, 0xffff
 567: s_mov_b32 s0, s4
 568: buffer_load_dwordx4 v[172:175], v152, s[0:3], 0 offen
 569: buffer_load_dwordx4 v[176:179], v153, s[0:3], 0 offen
 570: buffer_load_dwordx4 v[180:183], v154, s[0:3], 0 offen
 571: buffer_load_dwordx4 v[184:187], v155, s[0:3], 0 offen
 572: ds_read2st64_b64 v[188:191], v163 offset1:8
 573: ds_read2st64_b64 v[192:195], v163 offset0:16 offset1:24
 574: ds_read2st64_b64 v[196:199], v163 offset0:32 offset1:40
 575: ds_read2st64_b64 v[200:203], v163 offset0:48 offset1:56
 576: ds_read2st64_b64 v[204:207], v165 offset0:16 offset1:32
 577: ds_read_b64 v[208:209], v164 offset:32768
 578: ds_read_b64 v[210:211], v165 offset:24576
 579: s_waitcnt lgkmcnt(0)
 580: s_barrier
 581: s_waitcnt lgkmcnt(0)
 582: v_mfma_f32_16x16x16_f16 v[126:129], v[208:209], v[188:189], v[126:129]
 583: s_setprio 1
 584: v_mfma_f32_16x16x16_f16 v[122:125], v[204:205], v[188:189], v[122:125]
 585: v_mfma_f32_16x16x16_f16 v[118:121], v[206:207], v[188:189], v[118:121]
 586: v_mfma_f32_16x16x16_f16 v[114:117], v[210:211], v[188:189], v[114:117]
 587: v_mfma_f32_16x16x16_f16 v[110:113], v[208:209], v[190:191], v[110:113]
 588: v_mfma_f32_16x16x16_f16 v[106:109], v[204:205], v[190:191], v[106:109]
 589: v_mfma_f32_16x16x16_f16 v[102:105], v[206:207], v[190:191], v[102:105]
 590: v_mfma_f32_16x16x16_f16 v[98:101], v[210:211], v[190:191], v[98:101]
 591: v_mfma_f32_16x16x16_f16 v[94:97], v[208:209], v[192:193], v[94:97]
 592: v_mfma_f32_16x16x16_f16 v[90:93], v[204:205], v[192:193], v[90:93]
 593: v_mfma_f32_16x16x16_f16 v[86:89], v[206:207], v[192:193], v[86:89]
 594: v_mfma_f32_16x16x16_f16 v[82:85], v[210:211], v[192:193], v[82:85]
 595: v_mfma_f32_16x16x16_f16 v[78:81], v[208:209], v[194:195], v[78:81]
 596: v_mfma_f32_16x16x16_f16 v[74:77], v[204:205], v[194:195], v[74:77]
 597: v_mfma_f32_16x16x16_f16 v[70:73], v[206:207], v[194:195], v[70:73]
 598: v_mfma_f32_16x16x16_f16 v[66:69], v[210:211], v[194:195], v[66:69]
 599: v_mfma_f32_16x16x16_f16 v[62:65], v[208:209], v[196:197], v[62:65]
 600: v_mfma_f32_16x16x16_f16 v[58:61], v[204:205], v[196:197], v[58:61]
 601: v_mfma_f32_16x16x16_f16 v[54:57], v[206:207], v[196:197], v[54:57]
 602: v_mfma_f32_16x16x16_f16 v[50:53], v[210:211], v[196:197], v[50:53]
 603: v_mfma_f32_16x16x16_f16 v[46:49], v[208:209], v[198:199], v[46:49]
 604: v_mfma_f32_16x16x16_f16 v[42:45], v[204:205], v[198:199], v[42:45]
 605: v_mfma_f32_16x16x16_f16 v[38:41], v[206:207], v[198:199], v[38:41]
 606: v_mfma_f32_16x16x16_f16 v[34:37], v[210:211], v[198:199], v[34:37]
 607: v_mfma_f32_16x16x16_f16 v[30:33], v[208:209], v[200:201], v[30:33]
 608: v_mfma_f32_16x16x16_f16 v[26:29], v[204:205], v[200:201], v[26:29]
 609: v_mfma_f32_16x16x16_f16 v[22:25], v[206:207], v[200:201], v[22:25]
 610: v_mfma_f32_16x16x16_f16 v[18:21], v[210:211], v[200:201], v[18:21]
 611: v_mfma_f32_16x16x16_f16 v[14:17], v[208:209], v[202:203], v[14:17]
 612: v_mfma_f32_16x16x16_f16 v[10:13], v[204:205], v[202:203], v[10:13]
 613: v_mfma_f32_16x16x16_f16 v[6:9], v[206:207], v[202:203], v[6:9]
 614: v_mfma_f32_16x16x16_f16 v[2:5], v[210:211], v[202:203], v[2:5]
 615: s_setprio 0
 616: s_barrier
 617: s_waitcnt lgkmcnt(0)
 618: ds_read2st64_b64 v[188:191], v166 offset1:8
 619: ds_read2st64_b64 v[192:195], v166 offset0:16 offset1:24
 620: ds_read2st64_b64 v[196:199], v166 offset0:32 offset1:40
 621: ds_read2st64_b64 v[200:203], v166 offset0:48 offset1:56
 622: ds_read2st64_b64 v[204:207], v168 offset0:16 offset1:32
 623: ds_read2st64_b64 v[208:211], v169 offset1:8
 624: ds_read2st64_b64 v[212:215], v169 offset0:16 offset1:24
 625: ds_read2st64_b64 v[216:219], v169 offset0:32 offset1:40
 626: ds_read2st64_b64 v[220:223], v169 offset0:48 offset1:56
 627: ds_read2st64_b64 v[224:227], v171 offset0:16 offset1:32
 628: ds_read_b64 v[228:229], v167 offset:32768
 629: ds_read_b64 v[230:231], v168 offset:24576
 630: ds_read_b64 v[232:233], v170 offset:32768
 631: ds_read_b64 v[234:235], v171 offset:24576
 632: s_waitcnt lgkmcnt(0)
 633: s_barrier
 634: s_waitcnt lgkmcnt(0)
 635: v_mfma_f32_16x16x16_f16 v[126:129], v[228:229], v[188:189], v[126:129]
 636: s_setprio 1
 637: v_mfma_f32_16x16x16_f16 v[122:125], v[204:205], v[188:189], v[122:125]
 638: v_mfma_f32_16x16x16_f16 v[118:121], v[206:207], v[188:189], v[118:121]
 639: v_mfma_f32_16x16x16_f16 v[114:117], v[230:231], v[188:189], v[114:117]
 640: v_mfma_f32_16x16x16_f16 v[110:113], v[228:229], v[190:191], v[110:113]
 641: v_mfma_f32_16x16x16_f16 v[106:109], v[204:205], v[190:191], v[106:109]
 642: v_mfma_f32_16x16x16_f16 v[102:105], v[206:207], v[190:191], v[102:105]
 643: v_mfma_f32_16x16x16_f16 v[98:101], v[230:231], v[190:191], v[98:101]
 644: v_mfma_f32_16x16x16_f16 v[94:97], v[228:229], v[192:193], v[94:97]
 645: v_mfma_f32_16x16x16_f16 v[90:93], v[204:205], v[192:193], v[90:93]
 646: v_mfma_f32_16x16x16_f16 v[86:89], v[206:207], v[192:193], v[86:89]
 647: v_mfma_f32_16x16x16_f16 v[82:85], v[230:231], v[192:193], v[82:85]
 648: v_mfma_f32_16x16x16_f16 v[78:81], v[228:229], v[194:195], v[78:81]
 649: v_mfma_f32_16x16x16_f16 v[74:77], v[204:205], v[194:195], v[74:77]
 650: v_mfma_f32_16x16x16_f16 v[70:73], v[206:207], v[194:195], v[70:73]
 651: v_mfma_f32_16x16x16_f16 v[66:69], v[230:231], v[194:195], v[66:69]
 652: v_mfma_f32_16x16x16_f16 v[62:65], v[228:229], v[196:197], v[62:65]
 653: v_mfma_f32_16x16x16_f16 v[58:61], v[204:205], v[196:197], v[58:61]
 654: v_mfma_f32_16x16x16_f16 v[54:57], v[206:207], v[196:197], v[54:57]
 655: v_mfma_f32_16x16x16_f16 v[50:53], v[230:231], v[196:197], v[50:53]
 656: v_mfma_f32_16x16x16_f16 v[46:49], v[228:229], v[198:199], v[46:49]
 657: v_mfma_f32_16x16x16_f16 v[42:45], v[204:205], v[198:199], v[42:45]
 658: v_mfma_f32_16x16x16_f16 v[38:41], v[206:207], v[198:199], v[38:41]
 659: v_mfma_f32_16x16x16_f16 v[34:37], v[230:231], v[198:199], v[34:37]
 660: v_mfma_f32_16x16x16_f16 v[30:33], v[228:229], v[200:201], v[30:33]
 661: v_mfma_f32_16x16x16_f16 v[26:29], v[204:205], v[200:201], v[26:29]
 662: v_mfma_f32_16x16x16_f16 v[22:25], v[206:207], v[200:201], v[22:25]
 663: v_mfma_f32_16x16x16_f16 v[18:21], v[230:231], v[200:201], v[18:21]
 664: v_mfma_f32_16x16x16_f16 v[14:17], v[228:229], v[202:203], v[14:17]
 665: v_mfma_f32_16x16x16_f16 v[10:13], v[204:205], v[202:203], v[10:13]
 666: v_mfma_f32_16x16x16_f16 v[6:9], v[206:207], v[202:203], v[6:9]
 667: v_mfma_f32_16x16x16_f16 v[2:5], v[230:231], v[202:203], v[2:5]
 668: s_setprio 0
 669: s_barrier
 670: s_waitcnt lgkmcnt(0)
 671: s_waitcnt vmcnt(7)
 672: ds_write_b64 v156, v[134:135]
 673: s_waitcnt vmcnt(6)
 674: ds_write_b64 v156, v[142:143] offset:8192
 675: ds_write2st64_b64 v157, v[136:137], v[144:145] offset1:16
 676: s_waitcnt vmcnt(5)
 677: ds_write_b64 v156, v[130:131] offset:16384
 678: s_waitcnt vmcnt(4)
 679: ds_write_b64 v156, v[138:139] offset:24576
 680: ds_write2st64_b64 v157, v[132:133], v[140:141] offset0:32 offset1:48
 681: s_waitcnt vmcnt(3)
 682: ds_write_b64 v156, v[172:173] offset:32768
 683: ds_write_b64 v157, v[174:175] offset:32768
 684: s_waitcnt vmcnt(2)
 685: ds_write_b64 v158, v[176:177] offset:8192
 686: s_waitcnt vmcnt(1)
 687: ds_write_b64 v158, v[180:181] offset:16384
 688: ds_write2st64_b64 v159, v[178:179], v[182:183] offset0:16 offset1:32
 689: s_waitcnt vmcnt(0)
 690: ds_write_b64 v158, v[184:185] offset:24576
 691: ds_write_b64 v159, v[186:187] offset:24576
 692: s_waitcnt lgkmcnt(0)
 693: s_barrier
 694: s_waitcnt lgkmcnt(0)
 695: v_mfma_f32_16x16x16_f16 v[126:129], v[232:233], v[208:209], v[126:129]
 696: s_setprio 1
 697: v_mfma_f32_16x16x16_f16 v[122:125], v[224:225], v[208:209], v[122:125]
 698: v_mfma_f32_16x16x16_f16 v[118:121], v[226:227], v[208:209], v[118:121]
 699: v_mfma_f32_16x16x16_f16 v[114:117], v[234:235], v[208:209], v[114:117]
 700: v_mfma_f32_16x16x16_f16 v[110:113], v[232:233], v[210:211], v[110:113]
 701: v_mfma_f32_16x16x16_f16 v[106:109], v[224:225], v[210:211], v[106:109]
 702: v_mfma_f32_16x16x16_f16 v[102:105], v[226:227], v[210:211], v[102:105]
 703: v_mfma_f32_16x16x16_f16 v[98:101], v[234:235], v[210:211], v[98:101]
 704: v_mfma_f32_16x16x16_f16 v[94:97], v[232:233], v[212:213], v[94:97]
 705: v_mfma_f32_16x16x16_f16 v[90:93], v[224:225], v[212:213], v[90:93]
 706: v_mfma_f32_16x16x16_f16 v[86:89], v[226:227], v[212:213], v[86:89]
 707: v_mfma_f32_16x16x16_f16 v[82:85], v[234:235], v[212:213], v[82:85]
 708: v_mfma_f32_16x16x16_f16 v[78:81], v[232:233], v[214:215], v[78:81]
 709: v_mfma_f32_16x16x16_f16 v[74:77], v[224:225], v[214:215], v[74:77]
 710: v_mfma_f32_16x16x16_f16 v[70:73], v[226:227], v[214:215], v[70:73]
 711: v_mfma_f32_16x16x16_f16 v[66:69], v[234:235], v[214:215], v[66:69]
 712: v_mfma_f32_16x16x16_f16 v[62:65], v[232:233], v[216:217], v[62:65]
 713: v_mfma_f32_16x16x16_f16 v[58:61], v[224:225], v[216:217], v[58:61]
 714: v_mfma_f32_16x16x16_f16 v[54:57], v[226:227], v[216:217], v[54:57]
 715: v_mfma_f32_16x16x16_f16 v[50:53], v[234:235], v[216:217], v[50:53]
 716: v_mfma_f32_16x16x16_f16 v[46:49], v[232:233], v[218:219], v[46:49]
 717: v_mfma_f32_16x16x16_f16 v[42:45], v[224:225], v[218:219], v[42:45]
 718: v_mfma_f32_16x16x16_f16 v[38:41], v[226:227], v[218:219], v[38:41]
 719: v_mfma_f32_16x16x16_f16 v[34:37], v[234:235], v[218:219], v[34:37]
 720: v_mfma_f32_16x16x16_f16 v[30:33], v[232:233], v[220:221], v[30:33]
 721: v_mfma_f32_16x16x16_f16 v[26:29], v[224:225], v[220:221], v[26:29]
 722: v_mfma_f32_16x16x16_f16 v[22:25], v[226:227], v[220:221], v[22:25]
 723: v_mfma_f32_16x16x16_f16 v[18:21], v[234:235], v[220:221], v[18:21]
 724: v_mfma_f32_16x16x16_f16 v[14:17], v[232:233], v[222:223], v[14:17]
 725: v_mfma_f32_16x16x16_f16 v[10:13], v[224:225], v[222:223], v[10:13]
 726: v_mfma_f32_16x16x16_f16 v[6:9], v[226:227], v[222:223], v[6:9]
 727: v_mfma_f32_16x16x16_f16 v[2:5], v[234:235], v[222:223], v[2:5]
 728: s_setprio 0
 729: s_barrier
 730: s_waitcnt lgkmcnt(0)
 731: s_add_u32 s4, s4, 0x80
 732: s_addc_u32 s5, s5, 0
 733: s_add_u32 s11, s11, 0x80
 734: s_addc_u32 s16, s16, 0
 735: s_add_i32 s17, s17, -1
 736: s_cmp_lg_u32 s17, 0
 737: s_cbranch_scc1 65131
 738: s_and_saveexec_b64 s[0:1], vcc
 739: s_cbranch_execz 1
 740: s_barrier
 741: s_or_b64 exec, exec, s[0:1]
 742: s_andn2_b64 vcc, exec, s[14:15]
 743: s_cbranch_vccnz 4
 744: v_and_b32_e32 v0, 12, v160
 745: v_or_b32_e32 v130, 48, v0
 746: v_lshlrev_b32_e32 v131, 2, v151
 747: v_and_or_b32 v1, v1, 48, v151
 748: v_xor_b32_e32 v130, v130, v131
 749: v_lshlrev_b32_e32 v1, 6, v1
 750: v_or_b32_e32 v132, v1, v130
 751: v_lshlrev_b32_e32 v142, 1, v132
 752: v_or_b32_e32 v132, 32, v0
 753: v_xor_b32_e32 v132, v132, v131
 754: v_or_b32_e32 v133, v1, v132
 755: v_lshlrev_b32_e32 v138, 1, v133
 756: v_or_b32_e32 v133, 16, v0
 757: v_xor_b32_e32 v133, v133, v131
 758: v_or_b32_e32 v134, v1, v133
 759: v_xor_b32_e32 v131, v0, v131
 760: v_lshlrev_b32_e32 v134, 1, v134
 761: v_or_b32_e32 v0, v1, v131
 762: v_lshlrev_b32_e32 v135, 1, v0
 763: v_add_u32_e32 v0, 0, v138
 764: v_add_u32_e32 v136, 0, v134
 765: v_and_or_b32 v161, v150, 16, v151
 766: v_add_u32_e32 v137, 0, v135
 767: ds_read_b64 v[0:1], v0 offset:32768
 768: ds_read_b64 v[146:147], v136 offset:32768
 769: ds_read_b64 v[148:149], v137 offset:32768
 770: v_lshlrev_b32_e32 v136, 6, v161
 771: v_or_b32_e32 v131, v136, v131
 772: v_lshl_add_u32 v178, v131, 1, 0
 773: v_add_u32_e32 v131, 0, v142
 774: v_or_b32_e32 v133, v136, v133
 775: v_or_b32_e32 v132, v136, v132
 776: v_or_b32_e32 v130, v136, v130
 777: v_add_u32_e32 v152, s12, v135
 778: v_add_u32_e32 v154, s12, v134
 779: v_add_u32_e32 v158, s12, v142
 780: v_lshl_add_u32 v179, v133, 1, 0
 781: v_lshl_add_u32 v180, v132, 1, 0
 782: ds_read_b64 v[150:151], v131 offset:32768
 783: v_lshl_add_u32 v181, v130, 1, 0
 784: ds_read2st64_b64 v[134:137], v154 offset0:16 offset1:32
 785: ds_read_b64 v[154:155], v154 offset:24576
 786: ds_read2st64_b64 v[142:145], v158 offset0:16 offset1:32
 787: ds_read_b64 v[158:159], v158 offset:24576
 788: ds_read2st64_b64 v[130:133], v152 offset0:16 offset1:32
 789: ds_read_b64 v[152:153], v152 offset:24576
 790: ds_read2st64_b64 v[162:165], v178 offset1:8
 791: v_add_u32_e32 v156, s12, v138
 792: ds_read2st64_b64 v[166:169], v179 offset1:8
 793: ds_read2st64_b64 v[138:141], v156 offset0:16 offset1:32
 794: ds_read_b64 v[156:157], v156 offset:24576
 795: s_waitcnt lgkmcnt(3)
 796: v_mfma_f32_16x16x16_f16 v[126:129], v[148:149], v[162:163], v[126:129]
 797: ds_read2st64_b64 v[170:173], v180 offset1:8
 798: ds_read2st64_b64 v[174:177], v181 offset1:8
 799: v_mfma_f32_16x16x16_f16 v[122:125], v[130:131], v[162:163], v[122:125]
 800: v_mfma_f32_16x16x16_f16 v[118:121], v[132:133], v[162:163], v[118:121]
 801: v_mfma_f32_16x16x16_f16 v[114:117], v[152:153], v[162:163], v[114:117]
 802: v_mfma_f32_16x16x16_f16 v[110:113], v[148:149], v[164:165], v[110:113]
 803: v_mfma_f32_16x16x16_f16 v[106:109], v[130:131], v[164:165], v[106:109]
 804: v_mfma_f32_16x16x16_f16 v[102:105], v[132:133], v[164:165], v[102:105]
 805: v_mfma_f32_16x16x16_f16 v[98:101], v[152:153], v[164:165], v[98:101]
 806: ds_read2st64_b64 v[162:165], v178 offset0:16 offset1:24
 807: s_waitcnt lgkmcnt(5)
 808: v_mfma_f32_16x16x16_f16 v[126:129], v[146:147], v[166:167], v[126:129]
 809: v_mfma_f32_16x16x16_f16 v[122:125], v[134:135], v[166:167], v[122:125]
 810: v_mfma_f32_16x16x16_f16 v[118:121], v[136:137], v[166:167], v[118:121]
 811: v_mfma_f32_16x16x16_f16 v[114:117], v[154:155], v[166:167], v[114:117]
 812: v_mfma_f32_16x16x16_f16 v[110:113], v[146:147], v[168:169], v[110:113]
 813: v_mfma_f32_16x16x16_f16 v[106:109], v[134:135], v[168:169], v[106:109]
 814: v_mfma_f32_16x16x16_f16 v[102:105], v[136:137], v[168:169], v[102:105]
 815: v_mfma_f32_16x16x16_f16 v[98:101], v[154:155], v[168:169], v[98:101]
 816: ds_read2st64_b64 v[166:169], v179 offset0:16 offset1:24
 817: s_waitcnt lgkmcnt(1)
 818: v_mfma_f32_16x16x16_f16 v[94:97], v[148:149], v[162:163], v[94:97]
 819: v_mfma_f32_16x16x16_f16 v[90:93], v[130:131], v[162:163], v[90:93]
 820: v_mfma_f32_16x16x16_f16 v[86:89], v[132:133], v[162:163], v[86:89]
 821: v_mfma_f32_16x16x16_f16 v[82:85], v[152:153], v[162:163], v[82:85]
 822: v_mfma_f32_16x16x16_f16 v[78:81], v[148:149], v[164:165], v[78:81]
 823: v_mfma_f32_16x16x16_f16 v[74:77], v[130:131], v[164:165], v[74:77]
 824: v_mfma_f32_16x16x16_f16 v[70:73], v[132:133], v[164:165], v[70:73]
 825: v_mfma_f32_16x16x16_f16 v[66:69], v[152:153], v[164:165], v[66:69]
 826: ds_read2st64_b64 v[162:165], v178 offset0:32 offset1:40
 827: v_mfma_f32_16x16x16_f16 v[126:129], v[0:1], v[170:171], v[126:129]
 828: v_mfma_f32_16x16x16_f16 v[122:125], v[138:139], v[170:171], v[122:125]
 829: v_mfma_f32_16x16x16_f16 v[118:121], v[140:141], v[170:171], v[118:121]
 830: v_mfma_f32_16x16x16_f16 v[114:117], v[156:157], v[170:171], v[114:117]
 831: v_mfma_f32_16x16x16_f16 v[110:113], v[0:1], v[172:173], v[110:113]
 832: v_mfma_f32_16x16x16_f16 v[106:109], v[138:139], v[172:173], v[106:109]
 833: v_mfma_f32_16x16x16_f16 v[102:105], v[140:141], v[172:173], v[102:105]
 834: v_mfma_f32_16x16x16_f16 v[98:101], v[156:157], v[172:173], v[98:101]
 835: ds_read2st64_b64 v[170:173], v180 offset0:16 offset1:24
 836: s_waitcnt lgkmcnt(2)
 837: v_mfma_f32_16x16x16_f16 v[94:97], v[146:147], v[166:167], v[94:97]
 838: v_mfma_f32_16x16x16_f16 v[90:93], v[134:135], v[166:167], v[90:93]
 839: v_mfma_f32_16x16x16_f16 v[86:89], v[136:137], v[166:167], v[86:89]
 840: v_mfma_f32_16x16x16_f16 v[82:85], v[154:155], v[166:167], v[82:85]
 841: v_mfma_f32_16x16x16_f16 v[78:81], v[146:147], v[168:169], v[78:81]
 842: v_mfma_f32_16x16x16_f16 v[74:77], v[134:135], v[168:169], v[74:77]
 843: v_mfma_f32_16x16x16_f16 v[70:73], v[136:137], v[168:169], v[70:73]
 844: v_mfma_f32_16x16x16_f16 v[66:69], v[154:155], v[168:169], v[66:69]
 845: ds_read2st64_b64 v[166:169], v179 offset0:32 offset1:40
 846: s_waitcnt lgkmcnt(2)
 847: v_mfma_f32_16x16x16_f16 v[62:65], v[148:149], v[162:163], v[62:65]
 848: v_mfma_f32_16x16x16_f16 v[58:61], v[130:131], v[162:163], v[58:61]
 849: v_mfma_f32_16x16x16_f16 v[54:57], v[132:133], v[162:163], v[54:57]
 850: v_mfma_f32_16x16x16_f16 v[50:53], v[152:153], v[162:163], v[50:53]
 851: v_mfma_f32_16x16x16_f16 v[46:49], v[148:149], v[164:165], v[46:49]
 852: v_mfma_f32_16x16x16_f16 v[42:45], v[130:131], v[164:165], v[42:45]
 853: v_mfma_f32_16x16x16_f16 v[38:41], v[132:133], v[164:165], v[38:41]
 854: v_mfma_f32_16x16x16_f16 v[34:37], v[152:153], v[164:165], v[34:37]
 855: ds_read2st64_b64 v[162:165], v178 offset0:48 offset1:56
 856: s_waitcnt lgkmcnt(2)
 857: v_mfma_f32_16x16x16_f16 v[94:97], v[0:1], v[170:171], v[94:97]
 858: v_mfma_f32_16x16x16_f16 v[90:93], v[138:139], v[170:171], v[90:93]
 859: v_mfma_f32_16x16x16_f16 v[86:89], v[140:141], v[170:171], v[86:89]
 860: v_mfma_f32_16x16x16_f16 v[82:85], v[156:157], v[170:171], v[82:85]
 861: v_mfma_f32_16x16x16_f16 v[78:81], v[0:1], v[172:173], v[78:81]
 862: v_mfma_f32_16x16x16_f16 v[74:77], v[138:139], v[172:173], v[74:77]
 863: v_mfma_f32_16x16x16_f16 v[70:73], v[140:141], v[172:173], v[70:73]
 864: v_mfma_f32_16x16x16_f16 v[66:69], v[156:157], v[172:173], v[66:69]
 865: ds_read2st64_b64 v[170:173], v180 offset0:32 offset1:40
 866: s_waitcnt lgkmcnt(2)
 867: v_mfma_f32_16x16x16_f16 v[62:65], v[146:147], v[166:167], v[62:65]
 868: v_mfma_f32_16x16x16_f16 v[58:61], v[134:135], v[166:167], v[58:61]
 869: v_mfma_f32_16x16x16_f16 v[54:57], v[136:137], v[166:167], v[54:57]
 870: v_mfma_f32_16x16x16_f16 v[50:53], v[154:155], v[166:167], v[50:53]
 871: v_mfma_f32_16x16x16_f16 v[46:49], v[146:147], v[168:169], v[46:49]
 872: v_mfma_f32_16x16x16_f16 v[42:45], v[134:135], v[168:169], v[42:45]
 873: v_mfma_f32_16x16x16_f16 v[38:41], v[136:137], v[168:169], v[38:41]
 874: v_mfma_f32_16x16x16_f16 v[34:37], v[154:155], v[168:169], v[34:37]
 875: ds_read2st64_b64 v[166:169], v179 offset0:48 offset1:56
 876: s_waitcnt lgkmcnt(2)
 877: v_mfma_f32_16x16x16_f16 v[30:33], v[148:149], v[162:163], v[30:33]
 878: v_mfma_f32_16x16x16_f16 v[14:17], v[148:149], v[164:165], v[14:17]
 879: s_waitcnt lgkmcnt(1)
 880: v_mfma_f32_16x16x16_f16 v[62:65], v[0:1], v[170:171], v[62:65]
 881: v_mfma_f32_16x16x16_f16 v[58:61], v[138:139], v[170:171], v[58:61]
 882: v_mfma_f32_16x16x16_f16 v[54:57], v[140:141], v[170:171], v[54:57]
 883: v_mfma_f32_16x16x16_f16 v[50:53], v[156:157], v[170:171], v[50:53]
 884: v_mfma_f32_16x16x16_f16 v[46:49], v[0:1], v[172:173], v[46:49]
 885: v_mfma_f32_16x16x16_f16 v[42:45], v[138:139], v[172:173], v[42:45]
 886: v_mfma_f32_16x16x16_f16 v[38:41], v[140:141], v[172:173], v[38:41]
 887: v_mfma_f32_16x16x16_f16 v[34:37], v[156:157], v[172:173], v[34:37]
 888: ds_read2st64_b64 v[170:173], v180 offset0:48 offset1:56
 889: s_waitcnt lgkmcnt(1)
 890: v_mfma_f32_16x16x16_f16 v[30:33], v[146:147], v[166:167], v[30:33]
 891: v_mfma_f32_16x16x16_f16 v[14:17], v[146:147], v[168:169], v[14:17]
 892: v_mfma_f32_16x16x16_f16 v[126:129], v[150:151], v[174:175], v[126:129]
 893: v_mfma_f32_16x16x16_f16 v[122:125], v[142:143], v[174:175], v[122:125]
 894: v_mfma_f32_16x16x16_f16 v[118:121], v[144:145], v[174:175], v[118:121]
 895: v_mfma_f32_16x16x16_f16 v[114:117], v[158:159], v[174:175], v[114:117]
 896: v_mfma_f32_16x16x16_f16 v[110:113], v[150:151], v[176:177], v[110:113]
 897: v_mfma_f32_16x16x16_f16 v[106:109], v[142:143], v[176:177], v[106:109]
 898: v_mfma_f32_16x16x16_f16 v[102:105], v[144:145], v[176:177], v[102:105]
 899: v_mfma_f32_16x16x16_f16 v[98:101], v[158:159], v[176:177], v[98:101]
 900: ds_read2st64_b64 v[174:177], v181 offset0:16 offset1:24
 901: s_waitcnt lgkmcnt(1)
 902: v_mfma_f32_16x16x16_f16 v[30:33], v[0:1], v[170:171], v[30:33]
 903: v_mfma_f32_16x16x16_f16 v[26:29], v[130:131], v[162:163], v[26:29]
 904: v_mfma_f32_16x16x16_f16 v[22:25], v[132:133], v[162:163], v[22:25]
 905: v_mfma_f32_16x16x16_f16 v[18:21], v[152:153], v[162:163], v[18:21]
 906: v_mfma_f32_16x16x16_f16 v[14:17], v[0:1], v[172:173], v[14:17]
 907: v_mfma_f32_16x16x16_f16 v[10:13], v[130:131], v[164:165], v[10:13]
 908: v_mfma_f32_16x16x16_f16 v[6:9], v[132:133], v[164:165], v[6:9]
 909: v_mfma_f32_16x16x16_f16 v[0:3], v[152:153], v[164:165], v[2:5]
 910: s_waitcnt lgkmcnt(0)
 911: v_mfma_f32_16x16x16_f16 v[94:97], v[150:151], v[174:175], v[94:97]
 912: v_mfma_f32_16x16x16_f16 v[90:93], v[142:143], v[174:175], v[90:93]
 913: v_mfma_f32_16x16x16_f16 v[86:89], v[144:145], v[174:175], v[86:89]
 914: v_mfma_f32_16x16x16_f16 v[82:85], v[158:159], v[174:175], v[82:85]
 915: v_mfma_f32_16x16x16_f16 v[78:81], v[150:151], v[176:177], v[78:81]
 916: v_mfma_f32_16x16x16_f16 v[74:77], v[142:143], v[176:177], v[74:77]
 917: v_mfma_f32_16x16x16_f16 v[70:73], v[144:145], v[176:177], v[70:73]
 918: v_mfma_f32_16x16x16_f16 v[66:69], v[158:159], v[176:177], v[66:69]
 919: ds_read2st64_b64 v[174:177], v181 offset0:32 offset1:40
 920: v_mfma_f32_16x16x16_f16 v[26:29], v[134:135], v[166:167], v[26:29]
 921: v_mfma_f32_16x16x16_f16 v[22:25], v[136:137], v[166:167], v[22:25]
 922: v_mfma_f32_16x16x16_f16 v[18:21], v[154:155], v[166:167], v[18:21]
 923: v_mfma_f32_16x16x16_f16 v[10:13], v[134:135], v[168:169], v[10:13]
 924: v_mfma_f32_16x16x16_f16 v[6:9], v[136:137], v[168:169], v[6:9]
 925: v_mfma_f32_16x16x16_f16 v[0:3], v[154:155], v[168:169], v[0:3]
 926: s_waitcnt lgkmcnt(0)
 927: v_mfma_f32_16x16x16_f16 v[62:65], v[150:151], v[174:175], v[62:65]
 928: v_mfma_f32_16x16x16_f16 v[58:61], v[142:143], v[174:175], v[58:61]
 929: v_mfma_f32_16x16x16_f16 v[54:57], v[144:145], v[174:175], v[54:57]
 930: v_mfma_f32_16x16x16_f16 v[50:53], v[158:159], v[174:175], v[50:53]
 931: v_mfma_f32_16x16x16_f16 v[46:49], v[150:151], v[176:177], v[46:49]
 932: v_mfma_f32_16x16x16_f16 v[42:45], v[142:143], v[176:177], v[42:45]
 933: v_mfma_f32_16x16x16_f16 v[38:41], v[144:145], v[176:177], v[38:41]
 934: v_mfma_f32_16x16x16_f16 v[34:37], v[158:159], v[176:177], v[34:37]
 935: ds_read2st64_b64 v[174:177], v181 offset0:48 offset1:56
 936: v_mfma_f32_16x16x16_f16 v[26:29], v[138:139], v[170:171], v[26:29]
 937: v_mfma_f32_16x16x16_f16 v[22:25], v[140:141], v[170:171], v[22:25]
 938: v_mfma_f32_16x16x16_f16 v[18:21], v[156:157], v[170:171], v[18:21]
 939: v_mfma_f32_16x16x16_f16 v[10:13], v[138:139], v[172:173], v[10:13]
 940: v_mfma_f32_16x16x16_f16 v[6:9], v[140:141], v[172:173], v[6:9]
 941: v_mfma_f32_16x16x16_f16 v[0:3], v[156:157], v[172:173], v[0:3]
 942: s_waitcnt lgkmcnt(0)
 943: v_mfma_f32_16x16x16_f16 v[30:33], v[150:151], v[174:175], v[30:33]
 944: v_mfma_f32_16x16x16_f16 v[26:29], v[142:143], v[174:175], v[26:29]
 945: v_mfma_f32_16x16x16_f16 v[22:25], v[144:145], v[174:175], v[22:25]
 946: v_mfma_f32_16x16x16_f16 v[18:21], v[158:159], v[174:175], v[18:21]
 947: v_mfma_f32_16x16x16_f16 v[14:17], v[150:151], v[176:177], v[14:17]
 948: v_mfma_f32_16x16x16_f16 v[10:13], v[142:143], v[176:177], v[10:13]
 949: v_mfma_f32_16x16x16_f16 v[6:9], v[144:145], v[176:177], v[6:9]
 950: v_mfma_f32_16x16x16_f16 v[2:5], v[158:159], v[176:177], v[0:3]
 951: v_mul_lo_u32 v144, s13, v161
 952: s_lshl_b32 s1, s13, 5
 953: v_add_u32_e32 v145, s1, v144
 954: v_add_u32_e32 v146, s1, v145
 955: v_add_u32_e32 v147, s1, v146
 956: v_add_u32_e32 v148, s1, v147
 957: v_add_u32_e32 v149, s1, v148
 958: s_mul_i32 s0, s22, s13
 959: v_add_u32_e32 v150, s1, v149
 960: v_add_u32_e32 v151, s1, v150
 961: s_ashr_i32 s1, s0, 31
 962: s_lshl_b64 s[0:1], s[0:1], 1
 963: v_cvt_f16_f32_e32 v126, v126
 964: v_cvt_f16_f32_e32 v127, v127
 965: s_add_u32 s2, s6, s0
 966: v_and_b32_e32 v130, 60, v160
 967: v_cvt_f16_f32_e32 v128, v128
 968: v_cvt_f16_f32_e32 v129, v129
 969: s_addc_u32 s3, s7, s1
 970: s_ashr_i32 s11, s10, 31
 971: v_or_b32_e32 v131, 0xc0, v130
 972: v_or_b32_e32 v132, 0x80, v130
 973: v_or_b32_e32 v133, 64, v130
 974: v_or_b32_e32 v135, s10, v130
 975: v_or_b32_e32 v136, s22, v161
 976: v_cvt_f16_f32_e32 v122, v122
 977: v_cvt_f16_f32_e32 v123, v123
 978: s_lshl_b64 s[0:1], s[10:11], 1
 979: v_or_b32_e32 v0, s10, v131
 980: v_or_b32_e32 v1, s10, v132
 981: v_or_b32_e32 v134, s10, v133
 982: v_or_b32_e32 v137, 0xe0, v136
 983: v_or_b32_e32 v138, 0xc0, v136
 984: v_or_b32_e32 v139, 0xa0, v136
 985: v_or_b32_e32 v140, 0x80, v136
 986: v_or_b32_e32 v141, 0x60, v136
 987: v_or_b32_e32 v142, 64, v136
 988: v_or_b32_e32 v143, 32, v136
 989: v_cvt_f16_f32_e32 v124, v124
 990: v_cvt_f16_f32_e32 v125, v125
 991: s_add_u32 s24, s2, s0
 992: v_cmp_gt_i32_e64 s[22:23], s8, v136
 993: v_cmp_gt_i32_e64 s[6:7], s9, v135
 994: v_cvt_f16_f32_e32 v118, v118
 995: v_cvt_f16_f32_e32 v119, v119
 996: s_addc_u32 s25, s3, s1
 997: v_cmp_gt_i32_e64 s[18:19], s8, v143
 998: v_cmp_gt_i32_e64 s[16:17], s8, v142
 999: v_cmp_gt_i32_e64 s[14:15], s8, v141
1000: v_cmp_gt_i32_e64 s[12:13], s8, v140
1001: v_cmp_gt_i32_e64 s[10:11], s8, v139
1002: v_cmp_gt_i32_e64 s[20:21], s8, v138
1003: v_cmp_gt_i32_e32 vcc, s8, v137
1004: v_cmp_gt_i32_e64 s[4:5], s9, v134
1005: v_cmp_gt_i32_e64 s[2:3], s9, v1
1006: v_cmp_gt_i32_e64 s[0:1], s9, v0
1007: v_pack_b32_f16 v0, v126, v127
1008: v_add_lshl_u32 v126, v144, v130, 1
1009: v_bfrev_b32_e32 v127, 1
1010: s_and_b64 s[8:9], s[22:23], s[6:7]
1011: v_cvt_f16_f32_e32 v120, v120
1012: v_cvt_f16_f32_e32 v121, v121
1013: s_and_b32 s25, s25, 0xffff
1014: s_mov_b32 s27, 0x27000
1015: s_mov_b32 s26, 0x7ffffffe
1016: v_pack_b32_f16 v1, v128, v129
1017: v_cndmask_b32_e64 v126, v127, v126, s[8:9]
1018: v_cvt_f16_f32_e32 v114, v114
1019: v_cvt_f16_f32_e32 v115, v115
1020: buffer_store_dwordx2 v[0:1], v126, s[24:27], 0 offen
1021: v_pack_b32_f16 v0, v122, v123
1022: v_add_lshl_u32 v122, v144, v133, 1
1023: s_and_b64 s[8:9], s[22:23], s[4:5]
1024: v_cvt_f16_f32_e32 v116, v116
1025: v_cvt_f16_f32_e32 v117, v117
1026: v_pack_b32_f16 v1, v124, v125
1027: v_cndmask_b32_e64 v122, v127, v122, s[8:9]
1028: v_cvt_f16_f32_e32 v110, v110
1029: v_cvt_f16_f32_e32 v111, v111
1030: buffer_store_dwordx2 v[0:1], v122, s[24:27], 0 offen
1031: v_pack_b32_f16 v0, v118, v119
1032: v_add_lshl_u32 v118, v144, v132, 1
1033: s_and_b64 s[8:9], s[22:23], s[2:3]
1034: v_cvt_f16_f32_e32 v112, v112
1035: v_cvt_f16_f32_e32 v113, v113
1036: v_pack_b32_f16 v1, v120, v121
1037: v_cndmask_b32_e64 v118, v127, v118, s[8:9]
1038: v_cvt_f16_f32_e32 v106, v106
1039: v_cvt_f16_f32_e32 v107, v107
1040: buffer_store_dwordx2 v[0:1], v118, s[24:27], 0 offen
1041: v_pack_b32_f16 v0, v114, v115
1042: v_add_lshl_u32 v114, v144, v131, 1
1043: s_and_b64 s[8:9], s[22:23], s[0:1]
1044: v_cvt_f16_f32_e32 v108, v108
1045: v_cvt_f16_f32_e32 v109, v109
1046: v_pack_b32_f16 v1, v116, v117
1047: v_cndmask_b32_e64 v114, v127, v114, s[8:9]
1048: v_cvt_f16_f32_e32 v102, v102
1049: v_cvt_f16_f32_e32 v103, v103
1050: buffer_store_dwordx2 v[0:1], v114, s[24:27], 0 offen
1051: v_pack_b32_f16 v0, v110, v111
1052: v_add_lshl_u32 v110, v145, v130, 1
1053: s_and_b64 s[8:9], s[18:19], s[6:7]
1054: v_cvt_f16_f32_e32 v104, v104
1055: v_cvt_f16_f32_e32 v105, v105
1056: v_pack_b32_f16 v1, v112, v113
1057: v_cndmask_b32_e64 v110, v127, v110, s[8:9]
1058: v_cvt_f16_f32_e32 v98, v98
1059: v_cvt_f16_f32_e32 v99, v99
1060: buffer_store_dwordx2 v[0:1], v110, s[24:27], 0 offen
1061: v_pack_b32_f16 v0, v106, v107
1062: v_add_lshl_u32 v106, v145, v133, 1
1063: s_and_b64 s[8:9], s[18:19], s[4:5]
1064: v_cvt_f16_f32_e32 v100, v100
1065: v_cvt_f16_f32_e32 v101, v101
1066: v_pack_b32_f16 v1, v108, v109
1067: v_cndmask_b32_e64 v106, v127, v106, s[8:9]
1068: v_cvt_f16_f32_e32 v94, v94
1069: v_cvt_f16_f32_e32 v95, v95
1070: buffer_store_dwordx2 v[0:1], v106, s[24:27], 0 offen
1071: v_pack_b32_f16 v0, v102, v103
1072: v_add_lshl_u32 v102, v145, v132, 1
1073: s_and_b64 s[8:9], s[18:19], s[2:3]
1074: v_cvt_f16_f32_e32 v96, v96
1075: v_cvt_f16_f32_e32 v97, v97
1076: v_pack_b32_f16 v1, v104, v105
1077: v_cndmask_b32_e64 v102, v127, v102, s[8:9]
1078: v_cvt_f16_f32_e32 v90, v90
1079: v_cvt_f16_f32_e32 v91, v91
1080: buffer_store_dwordx2 v[0:1], v102, s[24:27], 0 offen
1081: v_pack_b32_f16 v0, v98, v99
1082: v_add_lshl_u32 v98, v145, v131, 1
1083: s_and_b64 s[8:9], s[18:19], s[0:1]
1084: v_cvt_f16_f32_e32 v92, v92
1085: v_cvt_f16_f32_e32 v93, v93
1086: v_pack_b32_f16 v1, v100, v101
1087: v_cndmask_b32_e64 v98, v127, v98, s[8:9]
1088: v_cvt_f16_f32_e32 v86, v86
1089: v_cvt_f16_f32_e32 v87, v87
1090: buffer_store_dwordx2 v[0:1], v98, s[24:27], 0 offen
1091: v_pack_b32_f16 v0, v94, v95
1092: v_add_lshl_u32 v94, v146, v130, 1
1093: s_and_b64 s[8:9], s[16:17], s[6:7]
1094: v_cvt_f16_f32_e32 v88, v88
1095: v_cvt_f16_f32_e32 v89, v89
1096: v_pack_b32_f16 v1, v96, v97
1097: v_cndmask_b32_e64 v94, v127, v94, s[8:9]
1098: v_cvt_f16_f32_e32 v82, v82
1099: v_cvt_f16_f32_e32 v83, v83
1100: buffer_store_dwordx2 v[0:1], v94, s[24:27], 0 offen
1101: v_pack_b32_f16 v0, v90, v91
1102: v_add_lshl_u32 v90, v146, v133, 1
1103: s_and_b64 s[8:9], s[16:17], s[4:5]
1104: v_cvt_f16_f32_e32 v84, v84
1105: v_cvt_f16_f32_e32 v85, v85
1106: v_pack_b32_f16 v1, v92, v93
1107: v_cndmask_b32_e64 v90, v127, v90, s[8:9]
1108: v_cvt_f16_f32_e32 v78, v78
1109: v_cvt_f16_f32_e32 v79, v79
1110: buffer_store_dwordx2 v[0:1], v90, s[24:27], 0 offen
1111: v_pack_b32_f16 v0, v86, v87
1112: v_add_lshl_u32 v86, v146, v132, 1
1113: s_and_b64 s[8:9], s[16:17], s[2:3]
1114: v_cvt_f16_f32_e32 v80, v80
1115: v_cvt_f16_f32_e32 v81, v81
1116: v_pack_b32_f16 v1, v88, v89
1117: v_cndmask_b32_e64 v86, v127, v86, s[8:9]
1118: v_cvt_f16_f32_e32 v74, v74
1119: v_cvt_f16_f32_e32 v75, v75
1120: buffer_store_dwordx2 v[0:1], v86, s[24:27], 0 offen
1121: v_pack_b32_f16 v0, v82, v83
1122: v_add_lshl_u32 v82, v146, v131, 1
1123: s_and_b64 s[8:9], s[16:17], s[0:1]
1124: v_cvt_f16_f32_e32 v76, v76
1125: v_cvt_f16_f32_e32 v77, v77
1126: v_pack_b32_f16 v1, v84, v85
1127: v_cndmask_b32_e64 v82, v127, v82, s[8:9]
1128: v_cvt_f16_f32_e32 v70, v70
1129: v_cvt_f16_f32_e32 v71, v71
1130: buffer_store_dwordx2 v[0:1], v82, s[24:27], 0 offen
1131: v_pack_b32_f16 v0, v78, v79
1132: v_add_lshl_u32 v78, v147, v130, 1
1133: s_and_b64 s[8:9], s[14:15], s[6:7]
1134: v_cvt_f16_f32_e32 v72, v72
1135: v_cvt_f16_f32_e32 v73, v73
1136: v_pack_b32_f16 v1, v80, v81
1137: v_cndmask_b32_e64 v78, v127, v78, s[8:9]
1138: v_cvt_f16_f32_e32 v66, v66
1139: v_cvt_f16_f32_e32 v67, v67
1140: buffer_store_dwordx2 v[0:1], v78, s[24:27], 0 offen
1141: v_pack_b32_f16 v0, v74, v75
1142: v_add_lshl_u32 v74, v147, v133, 1
1143: s_and_b64 s[8:9], s[14:15], s[4:5]
1144: v_cvt_f16_f32_e32 v68, v68
1145: v_cvt_f16_f32_e32 v69, v69
1146: v_pack_b32_f16 v1, v76, v77
1147: v_cndmask_b32_e64 v74, v127, v74, s[8:9]
1148: v_cvt_f16_f32_e32 v62, v62
1149: v_cvt_f16_f32_e32 v63, v63
1150: buffer_store_dwordx2 v[0:1], v74, s[24:27], 0 offen
1151: v_pack_b32_f16 v0, v70, v71
1152: v_add_lshl_u32 v70, v147, v132, 1
1153: s_and_b64 s[8:9], s[14:15], s[2:3]
1154: v_cvt_f16_f32_e32 v64, v64
1155: v_cvt_f16_f32_e32 v65, v65
1156: v_pack_b32_f16 v1, v72, v73
1157: v_cndmask_b32_e64 v70, v127, v70, s[8:9]
1158: v_cvt_f16_f32_e32 v58, v58
1159: v_cvt_f16_f32_e32 v59, v59
1160: buffer_store_dwordx2 v[0:1], v70, s[24:27], 0 offen
1161: v_pack_b32_f16 v0, v66, v67
1162: v_add_lshl_u32 v66, v147, v131, 1
1163: s_and_b64 s[8:9], s[14:15], s[0:1]
1164: v_cvt_f16_f32_e32 v60, v60
1165: v_cvt_f16_f32_e32 v61, v61
1166: v_pack_b32_f16 v1, v68, v69
1167: v_cndmask_b32_e64 v66, v127, v66, s[8:9]
1168: v_cvt_f16_f32_e32 v54, v54
1169: v_cvt_f16_f32_e32 v55, v55
1170: buffer_store_dwordx2 v[0:1], v66, s[24:27], 0 offen
1171: v_pack_b32_f16 v0, v62, v63
1172: v_add_lshl_u32 v62, v148, v130, 1
1173: s_and_b64 s[8:9], s[12:13], s[6:7]
1174: v_cvt_f16_f32_e32 v56, v56
1175: v_cvt_f16_f32_e32 v57, v57
1176: v_pack_b32_f16 v1, v64, v65
1177: v_cndmask_b32_e64 v62, v127, v62, s[8:9]
1178: v_cvt_f16_f32_e32 v50, v50
1179: v_cvt_f16_f32_e32 v51, v51
1180: buffer_store_dwordx2 v[0:1], v62, s[24:27], 0 offen
1181: v_pack_b32_f16 v0, v58, v59
1182: v_add_lshl_u32 v58, v148, v133, 1
1183: s_and_b64 s[8:9], s[12:13], s[4:5]
1184: v_cvt_f16_f32_e32 v52, v52
1185: v_cvt_f16_f32_e32 v53, v53
1186: v_pack_b32_f16 v1, v60, v61
1187: v_cndmask_b32_e64 v58, v127, v58, s[8:9]
1188: v_cvt_f16_f32_e32 v46, v46
1189: v_cvt_f16_f32_e32 v47, v47
1190: buffer_store_dwordx2 v[0:1], v58, s[24:27], 0 offen
1191: v_pack_b32_f16 v0, v54, v55
1192: v_add_lshl_u32 v54, v148, v132, 1
1193: s_and_b64 s[8:9], s[12:13], s[2:3]
1194: v_cvt_f16_f32_e32 v48, v48
1195: v_cvt_f16_f32_e32 v49, v49
1196: v_pack_b32_f16 v1, v56, v57
1197: v_cndmask_b32_e64 v54, v127, v54, s[8:9]
1198: v_cvt_f16_f32_e32 v42, v42
1199: v_cvt_f16_f32_e32 v43, v43
1200: buffer_store_dwordx2 v[0:1], v54, s[24:27], 0 offen
1201: v_pack_b32_f16 v0, v50, v51
1202: v_add_lshl_u32 v50, v148, v131, 1
1203: s_and_b64 s[8:9], s[12:13], s[0:1]
1204: v_cvt_f16_f32_e32 v44, v44
1205: v_cvt_f16_f32_e32 v45, v45
1206: v_pack_b32_f16 v1, v52, v53
1207: v_cndmask_b32_e64 v50, v127, v50, s[8:9]
1208: v_cvt_f16_f32_e32 v38, v38
1209: v_cvt_f16_f32_e32 v39, v39
1210: buffer_store_dwordx2 v[0:1], v50, s[24:27], 0 offen
1211: v_pack_b32_f16 v0, v46, v47
1212: v_add_lshl_u32 v46, v149, v130, 1
1213: s_and_b64 s[8:9], s[10:11], s[6:7]
1214: v_cvt_f16_f32_e32 v40, v40
1215: v_cvt_f16_f32_e32 v41, v41
1216: v_pack_b32_f16 v1, v48, v49
1217: v_cndmask_b32_e64 v46, v127, v46, s[8:9]
1218: v_cvt_f16_f32_e32 v34, v34
1219: v_cvt_f16_f32_e32 v35, v35
1220: buffer_store_dwordx2 v[0:1], v46, s[24:27], 0 offen
1221: v_pack_b32_f16 v0, v42, v43
1222: v_add_lshl_u32 v42, v149, v133, 1
1223: s_and_b64 s[8:9], s[10:11], s[4:5]
1224: v_cvt_f16_f32_e32 v36, v36
1225: v_cvt_f16_f32_e32 v37, v37
1226: v_pack_b32_f16 v1, v44, v45
1227: v_cndmask_b32_e64 v42, v127, v42, s[8:9]
1228: v_cvt_f16_f32_e32 v30, v30
1229: v_cvt_f16_f32_e32 v31, v31
1230: buffer_store_dwordx2 v[0:1], v42, s[24:27], 0 offen
1231: v_pack_b32_f16 v0, v38, v39
1232: v_add_lshl_u32 v38, v149, v132, 1
1233: s_and_b64 s[8:9], s[10:11], s[2:3]
1234: v_cvt_f16_f32_e32 v32, v32
1235: v_cvt_f16_f32_e32 v33, v33
1236: v_pack_b32_f16 v1, v40, v41
1237: v_cndmask_b32_e64 v38, v127, v38, s[8:9]
1238: v_cvt_f16_f32_e32 v26, v26
1239: v_cvt_f16_f32_e32 v27, v27
1240: buffer_store_dwordx2 v[0:1], v38, s[24:27], 0 offen
1241: v_pack_b32_f16 v0, v34, v35
1242: v_add_lshl_u32 v34, v149, v131, 1
1243: s_and_b64 s[8:9], s[10:11], s[0:1]
1244: v_cvt_f16_f32_e32 v28, v28
1245: v_cvt_f16_f32_e32 v29, v29
1246: v_pack_b32_f16 v1, v36, v37
1247: v_cndmask_b32_e64 v34, v127, v34, s[8:9]
1248: v_cvt_f16_f32_e32 v22, v22
1249: v_cvt_f16_f32_e32 v23, v23
1250: buffer_store_dwordx2 v[0:1], v34, s[24:27], 0 offen
1251: v_pack_b32_f16 v0, v30, v31
1252: v_add_lshl_u32 v30, v150, v130, 1
1253: s_and_b64 s[8:9], s[20:21], s[6:7]
1254: v_cvt_f16_f32_e32 v24, v24
1255: v_cvt_f16_f32_e32 v25, v25
1256: v_pack_b32_f16 v1, v32, v33
1257: v_cndmask_b32_e64 v30, v127, v30, s[8:9]
1258: v_cvt_f16_f32_e32 v18, v18
1259: v_cvt_f16_f32_e32 v19, v19
1260: buffer_store_dwordx2 v[0:1], v30, s[24:27], 0 offen
1261: v_pack_b32_f16 v0, v26, v27
1262: v_add_lshl_u32 v26, v150, v133, 1
1263: s_and_b64 s[8:9], s[20:21], s[4:5]
1264: v_cvt_f16_f32_e32 v20, v20
1265: v_cvt_f16_f32_e32 v21, v21
1266: v_pack_b32_f16 v1, v28, v29
1267: v_cndmask_b32_e64 v26, v127, v26, s[8:9]
1268: v_cvt_f16_f32_e32 v14, v14
1269: v_cvt_f16_f32_e32 v15, v15
1270: buffer_store_dwordx2 v[0:1], v26, s[24:27], 0 offen
1271: v_pack_b32_f16 v0, v22, v23
1272: v_add_lshl_u32 v22, v150, v132, 1
1273: s_and_b64 s[8:9], s[20:21], s[2:3]
1274: v_cvt_f16_f32_e32 v16, v16
1275: v_cvt_f16_f32_e32 v17, v17
1276: v_pack_b32_f16 v1, v24, v25
1277: v_cndmask_b32_e64 v22, v127, v22, s[8:9]
1278: v_cvt_f16_f32_e32 v10, v10
1279: v_cvt_f16_f32_e32 v11, v11
1280: buffer_store_dwordx2 v[0:1], v22, s[24:27], 0 offen
1281: v_pack_b32_f16 v0, v18, v19
1282: v_add_lshl_u32 v18, v150, v131, 1
1283: s_and_b64 s[8:9], s[20:21], s[0:1]
1284: v_cvt_f16_f32_e32 v12, v12
1285: v_cvt_f16_f32_e32 v13, v13
1286: v_pack_b32_f16 v1, v20, v21
1287: v_cndmask_b32_e64 v18, v127, v18, s[8:9]
1288: v_cvt_f16_f32_e32 v6, v6
1289: v_cvt_f16_f32_e32 v7, v7
1290: buffer_store_dwordx2 v[0:1], v18, s[24:27], 0 offen
1291: v_pack_b32_f16 v0, v14, v15
1292: v_add_lshl_u32 v14, v151, v130, 1
1293: s_and_b64 s[6:7], vcc, s[6:7]
1294: v_cvt_f16_f32_e32 v8, v8
1295: v_cvt_f16_f32_e32 v9, v9
1296: v_pack_b32_f16 v1, v16, v17
1297: v_cndmask_b32_e64 v14, v127, v14, s[6:7]
1298: v_cvt_f16_f32_e32 v2, v2
1299: v_cvt_f16_f32_e32 v3, v3
1300: buffer_store_dwordx2 v[0:1], v14, s[24:27], 0 offen
1301: v_pack_b32_f16 v0, v10, v11
1302: v_add_lshl_u32 v10, v151, v133, 1
1303: s_and_b64 s[4:5], vcc, s[4:5]
1304: v_cvt_f16_f32_e32 v4, v4
1305: v_cvt_f16_f32_e32 v5, v5
1306: v_pack_b32_f16 v1, v12, v13
1307: v_cndmask_b32_e64 v10, v127, v10, s[4:5]
1308: buffer_store_dwordx2 v[0:1], v10, s[24:27], 0 offen
1309: v_pack_b32_f16 v0, v6, v7
1310: v_add_lshl_u32 v6, v151, v132, 1
1311: s_and_b64 s[2:3], vcc, s[2:3]
1312: v_pack_b32_f16 v1, v8, v9
1313: v_cndmask_b32_e64 v6, v127, v6, s[2:3]
1314: buffer_store_dwordx2 v[0:1], v6, s[24:27], 0 offen
1315: v_pack_b32_f16 v0, v2, v3
1316: v_add_lshl_u32 v2, v151, v131, 1
1317: s_and_b64 vcc, vcc, s[0:1]
1318: v_pack_b32_f16 v1, v4, v5
1319: v_cndmask_b32_e32 v2, v127, v2, vcc
1320: buffer_store_dwordx2 v[0:1], v2, s[24:27], 0 offen
1321: s_barrier