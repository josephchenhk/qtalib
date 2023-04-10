# -*- coding: utf-8 -*-
# @Time    : 4/4/2023 5:20 pm
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: test_results.py

"""
Copyright (C) 2022 Joseph Chen - All Rights Reserved
You may use, distribute and modify this code under the terms of the JXW license, 
which unfortunately won't be written for another century.

You should have received a copy of the JXW license with this file. If not, 
please write to: josephchenhk@gmail.com
"""

import numpy as np

exp_obv_result = np.array(
    [0.,   218.,   272.,   312.,    51.,  -129.,  -287.,  -415.,
    -389.,  -253.,    20.,    20.,   -86.,   -34.,  -229.,  -151.,
    -439.,  -396.,  -238.,  -100.,  -172.,  -372.,  -254.,  -299.,
      -8.,  -174.,     5.,   -71.,   106.,  -166.,  -166.,   -74.,
    -318.,  -424.,  -593.,  -650.,  -611.,  -586.,  -882.,  -697.,
    -464.,  -635.,  -740.,  -590.,  -737.,  -438.,  -287.,  -385.,
    -289.,  -532.,  -457.,  -693.,  -693.,  -953.,  -881.,  -853.,
    -969., -1022.,  -881.,  -951.,  -739.,  -591.,  -858., -1056.,
    -857.,  -737.,  -737.,  -489.,  -618.,  -649.,  -462.,  -651.,
    -630.,  -688.,  -873.,  -680.,  -587.,  -876., -1170., -1448.,
   -1261., -1325., -1352., -1084.,  -806.,  -612.,  -436.,  -666.,
    -818.,  -704.,  -505.,  -323.,  -593.,  -351.,  -528.,  -747.,
    -881.,  -619.,  -619.,  -692.,  -555.,  -738.,  -907.,  -725.,
    -605.,  -851.,  -955., -1013., -1274., -1193., -1365., -1171.,
   -1066., -1238., -1006., -1213., -1213., -1074., -1130.,  -958.,
    -843.,  -843.,  -717.,  -779.,  -579.,  -797.,  -646.,  -459.,
    -409.,  -371.,  -436.,  -403.,  -680.,  -895.,  -678.,  -678.,
    -457.,  -720.,  -543.,  -476.,  -574.,  -642.,  -583.,  -333.,
    -617.,  -463.,  -589.,  -726.,  -529.,  -379.,  -457.,  -408.,
    -216.,   -39.,  -241.,  -519.,  -818.,  -745.,  -722.,  -470.,
    -628.,  -340.,  -395.,  -314.,  -432.,  -696.,  -658.,  -718.,
    -831.,  -773.,  -800.,  -743.,  -827.,  -762.,  -788.,  -751.,
    -518.,  -572.,  -325.,  -560.]
)

exp_wobv_result = np.array(
    [0.        ,    70.89430894,   100.44864827,   139.1999938 ,
    -121.8000062 ,  -301.8000062 ,  -400.5500062 ,  -493.47196627,
    -488.85384904,  -352.85384904,  -162.89361047,  -162.89361047,
    -224.16528677,  -172.16528677,  -304.1449822 ,  -245.05407311,
    -382.36039016,  -351.55179241,  -306.08416651,  -226.77382168,
    -275.34953383,  -415.79897203,  -312.54897203,  -357.54897203,
     -66.54897203,  -214.10452758,   -64.7600699 ,  -123.30107579,
       9.35417047,  -262.64582953,  -262.64582953,  -194.72987269,
    -438.72987269,  -472.11569946,  -540.44454043,  -575.14773678,
    -558.22799708,  -536.62972494,  -832.62972494,  -705.41950883,
    -609.33703461,  -679.70740498,  -784.70740498,  -665.77777168,
    -740.58693198,  -599.21577359,  -503.34275772,  -576.1843815 ,
    -493.73039377,  -535.55483439,  -463.1717609 ,  -620.50509423,
    -620.50509423,  -819.14685993,  -802.24545148,  -781.43355897,
    -838.29630407,  -891.29630407,  -781.99397849,  -838.98176682,
    -672.92432557,  -647.36301296,  -853.85953268, -1014.70437427,
    -815.70437427,  -764.69480998,  -764.69480998,  -723.08407173,
    -747.51588991,  -756.34779874,  -569.34779874,  -711.84265823,
    -693.43402708,  -750.90353927,  -922.19983557,  -834.07198169,
    -741.07198169, -1030.07198169, -1298.56513237, -1465.28177405,
   -1298.93633299, -1350.82822488, -1372.01392671, -1259.88003549,
   -1168.83200056, -1138.13579803,  -962.13579803, -1123.45844332,
   -1226.50929078, -1116.89390616,  -917.89390616,  -761.44691476,
   -1031.44691476,  -794.42438782,  -971.42438782, -1036.12158132,
   -1112.83913858,  -936.02932262,  -936.02932262, -1005.68581117,
    -868.68581117, -1026.89906766, -1194.22580033, -1044.80051298,
    -931.37896288, -1177.37896288, -1198.95572637, -1205.4872579 ,
   -1466.4872579 , -1385.4872579 , -1461.66086286, -1267.66086286,
   -1194.46694651, -1366.46694651, -1148.11400533, -1355.11400533,
   -1355.11400533, -1216.11400533, -1269.28907952, -1120.72861886,
   -1098.78205398, -1098.78205398,  -972.78205398, -1014.31315446,
    -930.56106065, -1148.56106065, -1044.42312962,  -990.0626645 ,
    -947.90414848,  -938.96297201,  -995.26218461,  -967.71627476,
   -1219.37104439, -1343.07645291, -1276.71559664, -1276.71559664,
   -1061.58285327, -1319.98896681, -1253.82074251, -1186.82074251,
   -1284.82074251, -1322.49387271, -1278.68199152, -1125.93250068,
   -1289.62126149, -1164.6950078 , -1252.58340349, -1300.82284011,
   -1141.60947804, -1033.69580898, -1097.69580898, -1058.90952929,
    -970.9370207 ,  -793.9370207 ,  -813.52964922,  -983.56023026,
   -1282.56023026, -1264.79867308, -1252.27598706, -1123.26574815,
   -1177.56128079,  -891.84699508,  -930.22801392,  -904.09898166,
    -982.76564833, -1117.11679337, -1079.11679337, -1128.0963852 ,
   -1201.05190888, -1148.21378694, -1169.39025753, -1112.39025753,
   -1148.94378234, -1096.26955219, -1116.09359911, -1090.6815112 ,
   -1011.02339154, -1057.95440729,  -872.14032884,  -987.90387564]
)