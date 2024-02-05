\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] + x[8] + x[9]
   + x[10] + x[11] + x[12] + x[13] + x[14] + x[15] + x[16] + x[17] + x[18]
   + x[19] + x[20] + x[21] + x[22] + x[23] + x[24] + y[0] + y[1]
Subject To
 FollowerHPR[0]: - x[0] - x[1] - y[1] <= -1
 FollowerHPR[1]: - x[0] - x[2] - y[0] <= -1
 FollowerHPR[2]: - x[3] - y[0] - y[1] <= -1
 FollowerHPR[3]: - x[2] - x[3] - x[4] <= -1
 FollowerHPR[4]: - x[1] - x[3] - x[5] <= -1
 FollowerHPR[5]: - x[1] - x[2] - x[6] <= -1
 FollowerHPR[6]: - x[5] - x[6] - y[0] <= -1
 FollowerHPR[7]: - x[4] - x[6] - y[1] <= -1
 FollowerHPR[8]: - x[0] - x[4] - x[5] <= -1
 FollowerHPR[9]: - x[1] - x[4] - y[0] <= -1
 FollowerHPR[10]: - x[2] - x[5] - y[1] <= -1
 FollowerHPR[11]: - x[0] - x[3] - x[6] <= -1
 FollowerHPR[12]: - x[8] - x[9] - x[10] <= -1
 FollowerHPR[13]: - x[7] - x[9] - x[11] <= -1
 FollowerHPR[14]: - x[7] - x[8] - x[12] <= -1
 FollowerHPR[15]: - x[11] - x[12] - x[13] <= -1
 FollowerHPR[16]: - x[10] - x[12] - x[14] <= -1
 FollowerHPR[17]: - x[10] - x[11] - x[15] <= -1
 FollowerHPR[18]: - x[7] - x[14] - x[15] <= -1
 FollowerHPR[19]: - x[8] - x[13] - x[15] <= -1
 FollowerHPR[20]: - x[9] - x[13] - x[14] <= -1
 FollowerHPR[21]: - x[7] - x[10] - x[13] <= -1
 FollowerHPR[22]: - x[8] - x[11] - x[14] <= -1
 FollowerHPR[23]: - x[9] - x[12] - x[15] <= -1
 FollowerHPR[24]: - x[17] - x[18] - x[19] <= -1
 FollowerHPR[25]: - x[16] - x[18] - x[20] <= -1
 FollowerHPR[26]: - x[16] - x[17] - x[21] <= -1
 FollowerHPR[27]: - x[20] - x[21] - x[22] <= -1
 FollowerHPR[28]: - x[19] - x[21] - x[23] <= -1
 FollowerHPR[29]: - x[19] - x[20] - x[24] <= -1
 FollowerHPR[30]: - x[16] - x[23] - x[24] <= -1
 FollowerHPR[31]: - x[17] - x[22] - x[24] <= -1
 FollowerHPR[32]: - x[18] - x[22] - x[23] <= -1
 FollowerHPR[33]: - x[16] - x[19] - x[22] <= -1
 FollowerHPR[34]: - x[17] - x[20] - x[23] <= -1
 FollowerHPR[35]: - x[18] - x[21] - x[24] <= -1
 FollowerHPR[36]: - x[7] - x[16] - y[0] <= -1
 FollowerHPR[37]: - x[8] - x[21] - y[0] <= -1
 FollowerHPR[38]: - x[9] - x[20] - y[0] <= -1
 FollowerHPR[39]: - x[10] - x[22] - y[0] <= -1
 FollowerHPR[40]: - x[11] - x[18] - y[0] <= -1
 FollowerHPR[41]: - x[12] - x[17] - y[0] <= -1
 FollowerHPR[42]: - x[13] - x[19] - y[0] <= -1
 FollowerHPR[43]: - x[14] - x[24] - y[0] <= -1
 FollowerHPR[44]: - x[15] - x[23] - y[0] <= -1
 FollowerHPR[45]: - x[7] - x[21] - y[1] <= -1
 FollowerHPR[46]: - x[8] - x[17] - y[1] <= -1
 FollowerHPR[47]: - x[9] - x[19] - y[1] <= -1
 FollowerHPR[48]: - x[10] - x[18] - y[1] <= -1
 FollowerHPR[49]: - x[11] - x[23] - y[1] <= -1
 FollowerHPR[50]: - x[12] - x[16] - y[1] <= -1
 FollowerHPR[51]: - x[13] - x[24] - y[1] <= -1
 FollowerHPR[52]: - x[14] - x[20] - y[1] <= -1
 FollowerHPR[53]: - x[15] - x[22] - y[1] <= -1
 FollowerHPR[54]: - x[0] - x[7] - x[20] <= -1
 FollowerHPR[55]: - x[0] - x[8] - x[19] <= -1
 FollowerHPR[56]: - x[0] - x[9] - x[18] <= -1
 FollowerHPR[57]: - x[0] - x[10] - x[17] <= -1
 FollowerHPR[58]: - x[0] - x[11] - x[16] <= -1
 FollowerHPR[59]: - x[0] - x[12] - x[24] <= -1
 FollowerHPR[60]: - x[0] - x[13] - x[23] <= -1
 FollowerHPR[61]: - x[0] - x[14] - x[22] <= -1
 FollowerHPR[62]: - x[0] - x[15] - x[21] <= -1
 FollowerHPR[63]: - x[1] - x[7] - x[22] <= -1
 FollowerHPR[64]: - x[1] - x[8] - x[18] <= -1
 FollowerHPR[65]: - x[1] - x[9] - x[17] <= -1
 FollowerHPR[66]: - x[1] - x[10] - x[19] <= -1
 FollowerHPR[67]: - x[1] - x[11] - x[24] <= -1
 FollowerHPR[68]: - x[1] - x[12] - x[23] <= -1
 FollowerHPR[69]: - x[1] - x[13] - x[16] <= -1
 FollowerHPR[70]: - x[1] - x[14] - x[21] <= -1
 FollowerHPR[71]: - x[1] - x[15] - x[20] <= -1
 FollowerHPR[72]: - x[2] - x[7] - x[18] <= -1
 FollowerHPR[73]: - x[2] - x[8] - x[23] <= -1
 FollowerHPR[74]: - x[2] - x[9] - x[16] <= -1
 FollowerHPR[75]: - x[2] - x[10] - x[24] <= -1
 FollowerHPR[76]: - x[2] - x[11] - x[20] <= -1
 FollowerHPR[77]: - x[2] - x[12] - x[22] <= -1
 FollowerHPR[78]: - x[2] - x[13] - x[21] <= -1
 FollowerHPR[79]: - x[2] - x[14] - x[17] <= -1
 FollowerHPR[80]: - x[2] - x[15] - x[19] <= -1
 FollowerHPR[81]: - x[3] - x[7] - x[17] <= -1
 FollowerHPR[82]: - x[3] - x[8] - x[16] <= -1
 FollowerHPR[83]: - x[3] - x[9] - x[24] <= -1
 FollowerHPR[84]: - x[3] - x[10] - x[23] <= -1
 FollowerHPR[85]: - x[3] - x[11] - x[22] <= -1
 FollowerHPR[86]: - x[3] - x[12] - x[21] <= -1
 FollowerHPR[87]: - x[3] - x[13] - x[20] <= -1
 FollowerHPR[88]: - x[3] - x[14] - x[19] <= -1
 FollowerHPR[89]: - x[3] - x[15] - x[18] <= -1
 FollowerHPR[90]: - x[4] - x[7] - x[19] <= -1
 FollowerHPR[91]: - x[4] - x[8] - x[24] <= -1
 FollowerHPR[92]: - x[4] - x[9] - x[23] <= -1
 FollowerHPR[93]: - x[4] - x[10] - x[16] <= -1
 FollowerHPR[94]: - x[4] - x[11] - x[21] <= -1
 FollowerHPR[95]: - x[4] - x[12] - x[20] <= -1
 FollowerHPR[96]: - x[4] - x[13] - x[22] <= -1
 FollowerHPR[97]: - x[4] - x[14] - x[18] <= -1
 FollowerHPR[98]: - x[4] - x[15] - x[17] <= -1
 FollowerHPR[99]: - x[5] - x[7] - x[24] <= -1
 FollowerHPR[100]: - x[5] - x[8] - x[20] <= -1
 FollowerHPR[101]: - x[5] - x[9] - x[22] <= -1
 FollowerHPR[102]: - x[5] - x[10] - x[21] <= -1
 FollowerHPR[103]: - x[5] - x[11] - x[17] <= -1
 FollowerHPR[104]: - x[5] - x[12] - x[19] <= -1
 FollowerHPR[105]: - x[5] - x[13] - x[18] <= -1
 FollowerHPR[106]: - x[5] - x[14] - x[23] <= -1
 FollowerHPR[107]: - x[5] - x[15] - x[16] <= -1
 FollowerHPR[108]: - x[6] - x[7] - x[23] <= -1
 FollowerHPR[109]: - x[6] - x[8] - x[22] <= -1
 FollowerHPR[110]: - x[6] - x[9] - x[21] <= -1
 FollowerHPR[111]: - x[6] - x[10] - x[20] <= -1
 FollowerHPR[112]: - x[6] - x[11] - x[19] <= -1
 FollowerHPR[113]: - x[6] - x[12] - x[18] <= -1
 FollowerHPR[114]: - x[6] - x[13] - x[17] <= -1
 FollowerHPR[115]: - x[6] - x[14] - x[16] <= -1
 FollowerHPR[116]: - x[6] - x[15] - x[24] <= -1
 FollowerHPR[117]: - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7]
   - x[8] - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16]
   - x[17] - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24] - y[0]
   - y[1] <= -13
 ValueFunction: - y[0] - y[1] + w[3-5-1] + w[root-1-1] + w[1-2-1] <= 0
 FlowRoot: w[root-sink-0] + w[root-3-0] + w[root-1-1] = 1
 FlowSink: - w[root-sink-0] - w[4-sink-0] - w[5-sink-0] - w[6-sink-0]
   - w[2-sink-0] = -1
 R121: - w[3-4-0] + w[4-sink-0] = 0
 R122: - w[root-3-0] + w[3-4-0] + w[3-5-1] = 0
 R123: - w[3-5-1] + w[5-sink-0] = 0
 R124: - w[1-6-0] + w[6-sink-0] = 0
 R125: - w[root-1-1] + w[1-6-0] + w[1-2-1] = 0
 R126: - w[1-2-1] + w[2-sink-0] = 0
 DualFeasFollower[root-sink-0]: pi[root] - pi[sink] <= 0
 DualFeasFollower[root-3-0]: pi[root] - pi[3] <= 0
 DualFeasFollower[3-4-0]: - pi[4] + pi[3] <= 0
 DualFeasFollower[3-5-1]: pi[3] - pi[5] <= -1
 DualFeasFollower[root-1-1]: pi[root] - pi[1] <= -1
 DualFeasFollower[1-6-0]: - pi[6] + pi[1] <= 0
 DualFeasFollower[1-2-1]: pi[1] - pi[2] <= -1
 DualFeasLeader[4-sink-0]: - pi[sink] + pi[4] - gamma[4-sink-0] <= 0
 DualFeasLeader[5-sink-0]: - pi[sink] + pi[5] - gamma[5-sink-0] <= 0
 DualFeasLeader[6-sink-0]: - pi[sink] + pi[6] - gamma[6-sink-0] <= 0
 DualFeasLeader[2-sink-0]: - pi[sink] + pi[2] - gamma[2-sink-0] <= 0
 StrongDualRoot: w[3-5-1] + w[root-1-1] + w[1-2-1] + pi[root] = 0
 StrongDualSink: pi[sink] = 0
 R140: gamma[4-sink-0] - 1e+06 alpha[4-sink-0] <= 0
 R141: gamma[5-sink-0] - 1e+06 alpha[5-sink-0] <= 0
 R142: gamma[6-sink-0] - 1e+06 alpha[6-sink-0] <= 0
 R143: gamma[2-sink-0] - 1e+06 alpha[2-sink-0] <= 0
 R144: alpha[4-sink-0] - beta[4-sink-0,0] - beta[4-sink-0,1]
   - beta[4-sink-0,2] - beta[4-sink-0,6] - beta[4-sink-0,7]
   - beta[4-sink-0,9] - beta[4-sink-0,10] - beta[4-sink-0,36]
   - beta[4-sink-0,37] - beta[4-sink-0,38] - beta[4-sink-0,39]
   - beta[4-sink-0,40] - beta[4-sink-0,41] - beta[4-sink-0,42]
   - beta[4-sink-0,43] - beta[4-sink-0,44] - beta[4-sink-0,45]
   - beta[4-sink-0,46] - beta[4-sink-0,47] - beta[4-sink-0,48]
   - beta[4-sink-0,49] - beta[4-sink-0,50] - beta[4-sink-0,51]
   - beta[4-sink-0,52] - beta[4-sink-0,53] - beta[4-sink-0,117] <= 0
 R145: alpha[5-sink-0] - beta[5-sink-0,0] - beta[5-sink-0,1]
   - beta[5-sink-0,2] - beta[5-sink-0,6] - beta[5-sink-0,7]
   - beta[5-sink-0,9] - beta[5-sink-0,10] - beta[5-sink-0,36]
   - beta[5-sink-0,37] - beta[5-sink-0,38] - beta[5-sink-0,39]
   - beta[5-sink-0,40] - beta[5-sink-0,41] - beta[5-sink-0,42]
   - beta[5-sink-0,43] - beta[5-sink-0,44] - beta[5-sink-0,45]
   - beta[5-sink-0,46] - beta[5-sink-0,47] - beta[5-sink-0,48]
   - beta[5-sink-0,49] - beta[5-sink-0,50] - beta[5-sink-0,51]
   - beta[5-sink-0,52] - beta[5-sink-0,53] - beta[5-sink-0,117] <= 0
 R146: alpha[6-sink-0] - beta[6-sink-0,0] - beta[6-sink-0,1]
   - beta[6-sink-0,2] - beta[6-sink-0,6] - beta[6-sink-0,7]
   - beta[6-sink-0,9] - beta[6-sink-0,10] - beta[6-sink-0,36]
   - beta[6-sink-0,37] - beta[6-sink-0,38] - beta[6-sink-0,39]
   - beta[6-sink-0,40] - beta[6-sink-0,41] - beta[6-sink-0,42]
   - beta[6-sink-0,43] - beta[6-sink-0,44] - beta[6-sink-0,45]
   - beta[6-sink-0,46] - beta[6-sink-0,47] - beta[6-sink-0,48]
   - beta[6-sink-0,49] - beta[6-sink-0,50] - beta[6-sink-0,51]
   - beta[6-sink-0,52] - beta[6-sink-0,53] - beta[6-sink-0,117] <= 0
 R147: alpha[2-sink-0] - beta[2-sink-0,0] - beta[2-sink-0,1]
   - beta[2-sink-0,2] - beta[2-sink-0,6] - beta[2-sink-0,7]
   - beta[2-sink-0,9] - beta[2-sink-0,10] - beta[2-sink-0,36]
   - beta[2-sink-0,37] - beta[2-sink-0,38] - beta[2-sink-0,39]
   - beta[2-sink-0,40] - beta[2-sink-0,41] - beta[2-sink-0,42]
   - beta[2-sink-0,43] - beta[2-sink-0,44] - beta[2-sink-0,45]
   - beta[2-sink-0,46] - beta[2-sink-0,47] - beta[2-sink-0,48]
   - beta[2-sink-0,49] - beta[2-sink-0,50] - beta[2-sink-0,51]
   - beta[2-sink-0,52] - beta[2-sink-0,53] - beta[2-sink-0,117] <= 0
 R148: - x[0] - x[1] - 1e+06 beta[4-sink-0,0] >= -1e+06
 R149: - x[0] - x[2] - 1e+06 beta[4-sink-0,1] >= -1e+06
 R150: - x[3] - 1e+06 beta[4-sink-0,2] >= -1e+06
 R151: - x[5] - x[6] - 1e+06 beta[4-sink-0,6] >= -1e+06
 R152: - x[4] - x[6] - 1e+06 beta[4-sink-0,7] >= -1e+06
 R153: - x[1] - x[4] - 1e+06 beta[4-sink-0,9] >= -1e+06
 R154: - x[2] - x[5] - 1e+06 beta[4-sink-0,10] >= -1e+06
 R155: - x[7] - x[16] - 1e+06 beta[4-sink-0,36] >= -1e+06
 R156: - x[8] - x[21] - 1e+06 beta[4-sink-0,37] >= -1e+06
 R157: - x[9] - x[20] - 1e+06 beta[4-sink-0,38] >= -1e+06
 R158: - x[10] - x[22] - 1e+06 beta[4-sink-0,39] >= -1e+06
 R159: - x[11] - x[18] - 1e+06 beta[4-sink-0,40] >= -1e+06
 R160: - x[12] - x[17] - 1e+06 beta[4-sink-0,41] >= -1e+06
 R161: - x[13] - x[19] - 1e+06 beta[4-sink-0,42] >= -1e+06
 R162: - x[14] - x[24] - 1e+06 beta[4-sink-0,43] >= -1e+06
 R163: - x[15] - x[23] - 1e+06 beta[4-sink-0,44] >= -1e+06
 R164: - x[7] - x[21] - 1e+06 beta[4-sink-0,45] >= -1e+06
 R165: - x[8] - x[17] - 1e+06 beta[4-sink-0,46] >= -1e+06
 R166: - x[9] - x[19] - 1e+06 beta[4-sink-0,47] >= -1e+06
 R167: - x[10] - x[18] - 1e+06 beta[4-sink-0,48] >= -1e+06
 R168: - x[11] - x[23] - 1e+06 beta[4-sink-0,49] >= -1e+06
 R169: - x[12] - x[16] - 1e+06 beta[4-sink-0,50] >= -1e+06
 R170: - x[13] - x[24] - 1e+06 beta[4-sink-0,51] >= -1e+06
 R171: - x[14] - x[20] - 1e+06 beta[4-sink-0,52] >= -1e+06
 R172: - x[15] - x[22] - 1e+06 beta[4-sink-0,53] >= -1e+06
 R173: - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8]
   - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16] - x[17]
   - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24]
   - 999988 beta[4-sink-0,117] >= -1e+06
 R174: - x[0] - x[1] - 1.000001e+06 beta[5-sink-0,0] >= -1e+06
 R175: - x[0] - x[2] - 1e+06 beta[5-sink-0,1] >= -1e+06
 R176: - x[3] - 1.000001e+06 beta[5-sink-0,2] >= -1e+06
 R177: - x[5] - x[6] - 1e+06 beta[5-sink-0,6] >= -1e+06
 R178: - x[4] - x[6] - 1.000001e+06 beta[5-sink-0,7] >= -1e+06
 R179: - x[1] - x[4] - 1e+06 beta[5-sink-0,9] >= -1e+06
 R180: - x[2] - x[5] - 1.000001e+06 beta[5-sink-0,10] >= -1e+06
 R181: - x[7] - x[16] - 1e+06 beta[5-sink-0,36] >= -1e+06
 R182: - x[8] - x[21] - 1e+06 beta[5-sink-0,37] >= -1e+06
 R183: - x[9] - x[20] - 1e+06 beta[5-sink-0,38] >= -1e+06
 R184: - x[10] - x[22] - 1e+06 beta[5-sink-0,39] >= -1e+06
 R185: - x[11] - x[18] - 1e+06 beta[5-sink-0,40] >= -1e+06
 R186: - x[12] - x[17] - 1e+06 beta[5-sink-0,41] >= -1e+06
 R187: - x[13] - x[19] - 1e+06 beta[5-sink-0,42] >= -1e+06
 R188: - x[14] - x[24] - 1e+06 beta[5-sink-0,43] >= -1e+06
 R189: - x[15] - x[23] - 1e+06 beta[5-sink-0,44] >= -1e+06
 R190: - x[7] - x[21] - 1.000001e+06 beta[5-sink-0,45] >= -1e+06
 R191: - x[8] - x[17] - 1.000001e+06 beta[5-sink-0,46] >= -1e+06
 R192: - x[9] - x[19] - 1.000001e+06 beta[5-sink-0,47] >= -1e+06
 R193: - x[10] - x[18] - 1.000001e+06 beta[5-sink-0,48] >= -1e+06
 R194: - x[11] - x[23] - 1.000001e+06 beta[5-sink-0,49] >= -1e+06
 R195: - x[12] - x[16] - 1.000001e+06 beta[5-sink-0,50] >= -1e+06
 R196: - x[13] - x[24] - 1.000001e+06 beta[5-sink-0,51] >= -1e+06
 R197: - x[14] - x[20] - 1.000001e+06 beta[5-sink-0,52] >= -1e+06
 R198: - x[15] - x[22] - 1.000001e+06 beta[5-sink-0,53] >= -1e+06
 R199: - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8]
   - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16] - x[17]
   - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24]
   - 999989 beta[5-sink-0,117] >= -1e+06
 R200: - x[0] - x[1] - 1e+06 beta[6-sink-0,0] >= -1e+06
 R201: - x[0] - x[2] - 1.000001e+06 beta[6-sink-0,1] >= -1e+06
 R202: - x[3] - 1.000001e+06 beta[6-sink-0,2] >= -1e+06
 R203: - x[5] - x[6] - 1.000001e+06 beta[6-sink-0,6] >= -1e+06
 R204: - x[4] - x[6] - 1e+06 beta[6-sink-0,7] >= -1e+06
 R205: - x[1] - x[4] - 1.000001e+06 beta[6-sink-0,9] >= -1e+06
 R206: - x[2] - x[5] - 1e+06 beta[6-sink-0,10] >= -1e+06
 R207: - x[7] - x[16] - 1.000001e+06 beta[6-sink-0,36] >= -1e+06
 R208: - x[8] - x[21] - 1.000001e+06 beta[6-sink-0,37] >= -1e+06
 R209: - x[9] - x[20] - 1.000001e+06 beta[6-sink-0,38] >= -1e+06
 R210: - x[10] - x[22] - 1.000001e+06 beta[6-sink-0,39] >= -1e+06
 R211: - x[11] - x[18] - 1.000001e+06 beta[6-sink-0,40] >= -1e+06
 R212: - x[12] - x[17] - 1.000001e+06 beta[6-sink-0,41] >= -1e+06
 R213: - x[13] - x[19] - 1.000001e+06 beta[6-sink-0,42] >= -1e+06
 R214: - x[14] - x[24] - 1.000001e+06 beta[6-sink-0,43] >= -1e+06
 R215: - x[15] - x[23] - 1.000001e+06 beta[6-sink-0,44] >= -1e+06
 R216: - x[7] - x[21] - 1e+06 beta[6-sink-0,45] >= -1e+06
 R217: - x[8] - x[17] - 1e+06 beta[6-sink-0,46] >= -1e+06
 R218: - x[9] - x[19] - 1e+06 beta[6-sink-0,47] >= -1e+06
 R219: - x[10] - x[18] - 1e+06 beta[6-sink-0,48] >= -1e+06
 R220: - x[11] - x[23] - 1e+06 beta[6-sink-0,49] >= -1e+06
 R221: - x[12] - x[16] - 1e+06 beta[6-sink-0,50] >= -1e+06
 R222: - x[13] - x[24] - 1e+06 beta[6-sink-0,51] >= -1e+06
 R223: - x[14] - x[20] - 1e+06 beta[6-sink-0,52] >= -1e+06
 R224: - x[15] - x[22] - 1e+06 beta[6-sink-0,53] >= -1e+06
 R225: - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8]
   - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16] - x[17]
   - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24]
   - 999989 beta[6-sink-0,117] >= -1e+06
 R226: - x[0] - x[1] - 1.000001e+06 beta[2-sink-0,0] >= -1e+06
 R227: - x[0] - x[2] - 1.000001e+06 beta[2-sink-0,1] >= -1e+06
 R228: - x[3] - 1.000002e+06 beta[2-sink-0,2] >= -1e+06
 R229: - x[5] - x[6] - 1.000001e+06 beta[2-sink-0,6] >= -1e+06
 R230: - x[4] - x[6] - 1.000001e+06 beta[2-sink-0,7] >= -1e+06
 R231: - x[1] - x[4] - 1.000001e+06 beta[2-sink-0,9] >= -1e+06
 R232: - x[2] - x[5] - 1.000001e+06 beta[2-sink-0,10] >= -1e+06
 R233: - x[7] - x[16] - 1.000001e+06 beta[2-sink-0,36] >= -1e+06
 R234: - x[8] - x[21] - 1.000001e+06 beta[2-sink-0,37] >= -1e+06
 R235: - x[9] - x[20] - 1.000001e+06 beta[2-sink-0,38] >= -1e+06
 R236: - x[10] - x[22] - 1.000001e+06 beta[2-sink-0,39] >= -1e+06
 R237: - x[11] - x[18] - 1.000001e+06 beta[2-sink-0,40] >= -1e+06
 R238: - x[12] - x[17] - 1.000001e+06 beta[2-sink-0,41] >= -1e+06
 R239: - x[13] - x[19] - 1.000001e+06 beta[2-sink-0,42] >= -1e+06
 R240: - x[14] - x[24] - 1.000001e+06 beta[2-sink-0,43] >= -1e+06
 R241: - x[15] - x[23] - 1.000001e+06 beta[2-sink-0,44] >= -1e+06
 R242: - x[7] - x[21] - 1.000001e+06 beta[2-sink-0,45] >= -1e+06
 R243: - x[8] - x[17] - 1.000001e+06 beta[2-sink-0,46] >= -1e+06
 R244: - x[9] - x[19] - 1.000001e+06 beta[2-sink-0,47] >= -1e+06
 R245: - x[10] - x[18] - 1.000001e+06 beta[2-sink-0,48] >= -1e+06
 R246: - x[11] - x[23] - 1.000001e+06 beta[2-sink-0,49] >= -1e+06
 R247: - x[12] - x[16] - 1.000001e+06 beta[2-sink-0,50] >= -1e+06
 R248: - x[13] - x[24] - 1.000001e+06 beta[2-sink-0,51] >= -1e+06
 R249: - x[14] - x[20] - 1.000001e+06 beta[2-sink-0,52] >= -1e+06
 R250: - x[15] - x[22] - 1.000001e+06 beta[2-sink-0,53] >= -1e+06
 R251: - x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8]
   - x[9] - x[10] - x[11] - x[12] - x[13] - x[14] - x[15] - x[16] - x[17]
   - x[18] - x[19] - x[20] - x[21] - x[22] - x[23] - x[24]
   - 999990 beta[2-sink-0,117] >= -1e+06
 R252: y[0] = 1
 R253: y[1] = 1
Bounds
 w[root-sink-0] <= 1
 w[root-3-0] <= 1
 w[3-4-0] <= 1
 w[4-sink-0] <= 1
 w[3-5-1] <= 1
 w[5-sink-0] <= 1
 w[root-1-1] <= 1
 w[1-6-0] <= 1
 w[6-sink-0] <= 1
 w[1-2-1] <= 1
 w[2-sink-0] <= 1
 pi[root] free
 pi[sink] free
 pi[4] free
 pi[3] free
 pi[5] free
 pi[6] free
 pi[1] free
 pi[2] free
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] y[0]
 y[1] alpha[4-sink-0] alpha[5-sink-0] alpha[6-sink-0] alpha[2-sink-0]
 beta[4-sink-0,0] beta[4-sink-0,1] beta[4-sink-0,2] beta[4-sink-0,6]
 beta[4-sink-0,7] beta[4-sink-0,9] beta[4-sink-0,10] beta[4-sink-0,36]
 beta[4-sink-0,37] beta[4-sink-0,38] beta[4-sink-0,39] beta[4-sink-0,40]
 beta[4-sink-0,41] beta[4-sink-0,42] beta[4-sink-0,43] beta[4-sink-0,44]
 beta[4-sink-0,45] beta[4-sink-0,46] beta[4-sink-0,47] beta[4-sink-0,48]
 beta[4-sink-0,49] beta[4-sink-0,50] beta[4-sink-0,51] beta[4-sink-0,52]
 beta[4-sink-0,53] beta[4-sink-0,117] beta[5-sink-0,0] beta[5-sink-0,1]
 beta[5-sink-0,2] beta[5-sink-0,6] beta[5-sink-0,7] beta[5-sink-0,9]
 beta[5-sink-0,10] beta[5-sink-0,36] beta[5-sink-0,37] beta[5-sink-0,38]
 beta[5-sink-0,39] beta[5-sink-0,40] beta[5-sink-0,41] beta[5-sink-0,42]
 beta[5-sink-0,43] beta[5-sink-0,44] beta[5-sink-0,45] beta[5-sink-0,46]
 beta[5-sink-0,47] beta[5-sink-0,48] beta[5-sink-0,49] beta[5-sink-0,50]
 beta[5-sink-0,51] beta[5-sink-0,52] beta[5-sink-0,53] beta[5-sink-0,117]
 beta[6-sink-0,0] beta[6-sink-0,1] beta[6-sink-0,2] beta[6-sink-0,6]
 beta[6-sink-0,7] beta[6-sink-0,9] beta[6-sink-0,10] beta[6-sink-0,36]
 beta[6-sink-0,37] beta[6-sink-0,38] beta[6-sink-0,39] beta[6-sink-0,40]
 beta[6-sink-0,41] beta[6-sink-0,42] beta[6-sink-0,43] beta[6-sink-0,44]
 beta[6-sink-0,45] beta[6-sink-0,46] beta[6-sink-0,47] beta[6-sink-0,48]
 beta[6-sink-0,49] beta[6-sink-0,50] beta[6-sink-0,51] beta[6-sink-0,52]
 beta[6-sink-0,53] beta[6-sink-0,117] beta[2-sink-0,0] beta[2-sink-0,1]
 beta[2-sink-0,2] beta[2-sink-0,6] beta[2-sink-0,7] beta[2-sink-0,9]
 beta[2-sink-0,10] beta[2-sink-0,36] beta[2-sink-0,37] beta[2-sink-0,38]
 beta[2-sink-0,39] beta[2-sink-0,40] beta[2-sink-0,41] beta[2-sink-0,42]
 beta[2-sink-0,43] beta[2-sink-0,44] beta[2-sink-0,45] beta[2-sink-0,46]
 beta[2-sink-0,47] beta[2-sink-0,48] beta[2-sink-0,49] beta[2-sink-0,50]
 beta[2-sink-0,51] beta[2-sink-0,52] beta[2-sink-0,53] beta[2-sink-0,117]
End
